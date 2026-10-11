import torch
import torch.nn.functional as F
import os
import logging

from tqdm import trange, tqdm
from transformers import BertTokenizer
from losses import loss_map
from utils.functions import save_model, restore_model
from utils.efficiency_recorder import NullEfficiencyRecorder
from .utils import *  # set_optimizer, view_generator, get_pseudo_dataloader

from backbones.base import freeze_bert_parameters


# UMC-ICLA 预训练管理器类，用于管理模型的预训练过程。
class PretrainUMCICLAManager:

    def __init__(self, args, data, model):
        # 构造函数，传入参数 args、数据 data 和模型 model

        # 获取日志记录器
        self.logger = logging.getLogger(args.logger_name)

        # 获取设备和模型
        self.device, self.model = model.device, model.model
        self.efficiency_recorder = (
            getattr(args, 'efficiency_recorder', None) or NullEfficiencyRecorder()
        )
        if args.freeze_pretrain_bert_parameters:
            self.logger.info('Freeze all parameters but the last layer for efficiency')
            self.model = freeze_bert_parameters(self.model, args.multimodal_method)

        self.efficiency_recorder.record_parameters(
            self.model, stage='pretrain_ready', paper_snapshot=False
        )

        # 设置优化器和学习率调度器
        self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr_pre)

        # 获取训练输出数据
        self.train_outputs = data.train_outputs

        # 设置对比损失函数
        self.contrast_criterion = loss_map['SupConLoss']
        # 设置虚拟对比损失函数
        self.virtual_contrast_loss = loss_map['SupConLoss']
        # 保存args参数
        self.args = args

        # 加载预训练的 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)
        # 创建视图生成器
        self.generator = view_generator(self.tokenizer, args)

        # 如果需要预训练，则调用 _train 方法
        if args.pretrain:
            self.logger.info('Pre-training start...')
            self._train(args)
            self.logger.info('Pre-training finished...')

        # 如果不需要预训练，则恢复预训练模型
        else:
            self.model = restore_model(self.model, os.path.join(args.model_output_path, 'pretrain'), self.device)

        # 将模型移动到 CPU
        self.model.to(torch.device('cpu'))
        # 清空 GPU 缓存
        torch.cuda.empty_cache()

    # 预训练方法
    # 预训练方法
    # 预训练方法
    def _train(self, args):

        # 获取伪数据和伪数据加载器
        pseudo_data, pseudo_dataloader = get_pseudo_dataloader(args, self.train_outputs, mode='pretrain')

        # 如果启用数据增强，创建增强数据加载器
        if hasattr(args, 'use_data_augmentation') and args.use_data_augmentation:
            from data.augmentation import get_augmented_dataloader
            _, augmented_dataloader = get_augmented_dataloader(args, self.train_outputs, self.tokenizer)

        # 遍历预训练轮数，显示进度条
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            with self.efficiency_recorder.segment(
                    'pretrain', 'train', epoch_stage=epoch + 1) as efficiency_segment:
                # 设置模型为训练模式
                self.model.train()
                # 步平均值供原日志使用；加权和供效率数据离线分析使用
                tr_loss, nb_tr_steps = 0, 0
                samples_seen = 0
                total_weighted_loss_sum = 0.0
                mm_weighted_loss_sum = 0.0
                virtual_weighted_loss_sum = 0.0
                virtual_samples_seen = 0

                # 遍历伪数据加载器，显示进度条
                for batch in tqdm(pseudo_dataloader, desc="Iteration"):
                    # 获取文本特征、视频特征、音频特征，并移动到指定设备
                    text_feats = batch['text_feats'].to(self.device)
                    video_feats = batch['video_feats'].to(self.device)
                    audio_feats = batch['audio_feats'].to(self.device)

                    # 启用梯度计算
                    with torch.set_grad_enabled(True):
                        mlp_output_a = self.model(
                            text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats,
                            mode='pretrain-mm')
                        mlp_output_b = self.model(
                            text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device),
                            mode='pretrain-mm')
                        mlp_output_c = self.model(
                            text_feats, video_feats, audio_feats, mode='pretrain-mm')

                        norm_mlp_output_a = F.normalize(mlp_output_a)
                        norm_mlp_output_b = F.normalize(mlp_output_b)
                        norm_mlp_output_c = F.normalize(mlp_output_c)

                        contrastive_logits = torch.cat((
                            norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1),
                            norm_mlp_output_c.unsqueeze(1)), dim=1)
                        loss_contrast_mm = self.contrast_criterion(
                            contrastive_logits, temperature=args.pretrain_temperature,
                            device=self.device)

                        # UMC-ICLA 的虚拟对比损失与多模态对比损失联合优化
                        contrast_loss = None
                        if hasattr(args, 'use_data_augmentation') and args.use_data_augmentation:
                            input_ids = torch.stack(
                                [torch.tensor(feat[0]) for feat in text_feats], dim=1
                            ).to(self.device)
                            attention_mask = torch.stack(
                                [torch.tensor(feat[1]) for feat in text_feats], dim=1
                            ).to(self.device)
                            _, contrast_loss = self.train_step_virtual(input_ids, attention_mask)

                        total_loss = (
                            loss_contrast_mm + contrast_loss
                            if contrast_loss is not None else loss_contrast_mm
                        )

                        self.optimizer.zero_grad()
                        total_loss.backward()

                        if args.grad_clip != -1.0:
                            torch.nn.utils.clip_grad_value_(
                                [param for param in self.model.parameters() if param.requires_grad],
                                args.grad_clip)

                        batch_size = int(text_feats.size(0))
                        total_loss_value = total_loss.item()
                        mm_loss_value = loss_contrast_mm.item()
                        tr_loss += total_loss_value
                        nb_tr_steps += 1
                        samples_seen += batch_size
                        total_weighted_loss_sum += total_loss_value * batch_size
                        mm_weighted_loss_sum += mm_loss_value * batch_size
                        if contrast_loss is not None:
                            virtual_weighted_loss_sum += contrast_loss.item() * batch_size
                            virtual_samples_seen += batch_size

                        self.optimizer.step()
                        self.scheduler.step()

                # 计算平均训练损失
                loss = tr_loss / nb_tr_steps

                # 将 ICLA 增强视图的额外计算纳入同一预训练 epoch 的时间和显存峰值
                aug_weighted_loss_sum = 0.0
                aug_samples_seen = 0
                nb_aug_steps = 0
                if (hasattr(args, 'use_data_augmentation') and args.use_data_augmentation
                        and 'augmented_dataloader' in locals()):
                    tr_aug_loss = 0
                    for batch_aug in tqdm(augmented_dataloader, desc='Augmentation'):
                        text_feats = batch_aug['text_feats'].to(self.device)
                        text_feats_aug = (
                            batch_aug['text_feats_aug'].to(self.device)
                            if 'text_feats_aug' in batch_aug else None
                        )

                        if text_feats_aug is not None:
                            input_ids = torch.stack(
                                [torch.tensor(feat[0]) for feat in text_feats_aug], dim=1
                            ).to(self.device)
                            attention_mask = torch.stack(
                                [torch.tensor(feat[1]) for feat in text_feats_aug], dim=1
                            ).to(self.device)
                            aug_losses, _ = self.train_step_virtual(input_ids, attention_mask)
                            aug_loss_value = aug_losses['loss']
                            aug_batch_size = int(text_feats_aug.size(0))
                            tr_aug_loss += aug_loss_value
                            nb_aug_steps += 1
                            aug_weighted_loss_sum += aug_loss_value * aug_batch_size
                            aug_samples_seen += aug_batch_size

                    if nb_aug_steps > 0:
                        tr_aug_loss /= nb_aug_steps
                        self.logger.info('Augmentation Training Loss: %f', tr_aug_loss)

                eval_results = {'train_loss': loss}
                self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
                for key in sorted(eval_results.keys()):
                    self.logger.info("  %s = %s", key, str(eval_results[key]))

            self.efficiency_recorder.record_epoch(
                stage='pretrain',
                epoch_stage=epoch + 1,
                segment=efficiency_segment,
                loss_components={
                    'multimodal_contrastive': (
                        mm_weighted_loss_sum / samples_seen if samples_seen else None
                    ),
                    'icla_virtual_contrastive': (
                        virtual_weighted_loss_sum / virtual_samples_seen
                        if virtual_samples_seen else None
                    ),
                    'icla_augmented_virtual': (
                        aug_weighted_loss_sum / aug_samples_seen
                        if aug_samples_seen else None
                    ),
                },
                samples_seen=samples_seen + aug_samples_seen,
                optimizer_steps=nb_tr_steps,
                learning_rates=[group['lr'] for group in self.optimizer.param_groups],
                loss_total_mean=(
                    total_weighted_loss_sum / samples_seen if samples_seen else None
                ),
            )

        # 如果需要保存模型，则保存预训练模型
        if args.save_model:
            pretrained_model_dir = os.path.join(args.model_output_path, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            with self.efficiency_recorder.segment('pretrain_checkpoint', 'checkpoint'):
                save_model(self.model, pretrained_model_dir)

    def train_step_virtual(self, input_ids, attention_mask):
        """虚拟数据增强的训练步骤"""
        # 设置模型为训练模式
        self.model.train()

        # 获取两个视图的嵌入
        embd1, embd2 = self.model([input_ids, attention_mask], None, None, mode='virtual')

        # 计算对比逻辑
        feat1, feat2 = self.model.contrast_logits(embd1, embd2)

        # 计算对比损失
        losses = {}
        # 将特征拼接为对比逻辑
        contrastive_logits = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        # 计算对比损失
        contrast_loss = self.virtual_contrast_loss(contrastive_logits, temperature=self.args.pretrain_temperature,
                                                   device=self.device)
        losses["loss"] = contrast_loss.item()

        # 返回contrast_loss，稍后与loss_contrast_mm一起优化
        return losses, contrast_loss
