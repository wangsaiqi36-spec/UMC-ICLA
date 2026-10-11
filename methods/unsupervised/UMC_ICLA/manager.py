import torch
import torch.nn.functional as F
import numpy as np
import logging
import os
import time
import copy

from sklearn.cluster import KMeans
from tqdm import trange, tqdm
from losses import loss_map
from utils.functions import save_model, restore_model, set_torch_seed
from transformers import BertTokenizer

from backbones.base import freeze_bert_parameters
from sklearn.neighbors import NearestNeighbors, KDTree
from utils.neighbor_dataset import NeighborsDataset
from torch.utils.data import DataLoader

from utils.metrics import clustering_score
from utils.efficiency_recorder import NullEfficiencyRecorder
from .pretrain import PretrainUMCICLAManager

from data.utils import get_dataloader
from .utils import *


class UMCICLAManager:
    # UMC-ICLA 管理器类，用于管理模型的训练和测试。

    def __init__(self, args, data, model):
        # 构造函数，传入参数 args、数据 data 和模型 model

        # 创建 UMC-ICLA 预训练管理器对象。
        pretrain_manager = PretrainUMCICLAManager(args, data, model)

        # 设置 PyTorch 的随机种子
        set_torch_seed(args.seed)

        self.logger = logging.getLogger(args.logger_name)
        # 获取设备和模型
        self.device, self.model = model.device, model.model
        self.efficiency_recorder = (
            getattr(args, 'efficiency_recorder', None) or NullEfficiencyRecorder()
        )

        # 获取多模态数据加载器
        mm_dataloader = get_dataloader(args, data.mm_data)
        # 分别获取训练和测试数据加载器
        self.train_dataloader, self.test_dataloader = mm_dataloader['train'], mm_dataloader['test']
        # 获取训练输出数据
        self.train_outputs = data.train_outputs

        # 设置交叉熵损失函数
        self.criterion = loss_map['CrossEntropyLoss']
        # 设置对比损失函数
        self.contrast_criterion = loss_map['SupConLoss']
        # 设置均方误差损失函数
        self.mse_criterion = loss_map['MSELoss']

        # 加载预训练的 BERT 分词器
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_model, do_lower_case=True)
        # 创建视图生成器
        self.generator = view_generator(self.tokenizer, args)
        # 初始化聚类中心为 None
        self.centroids = None

        # 预训练
        # 预训练
        # 预训练
        if args.pretrain:
            # 如果进行预训练，获取预训练模型
            self.pretrained_model = pretrain_manager.model

            # 获取标签数量
            self.num_labels = args.num_labels
            # 加载预训练模型
            self.load_pretrained_model(self.pretrained_model)

        else:
            # 如果不进行预训练，恢复预训练模型
            self.num_labels = args.num_labels
            self.pretrained_model = restore_model(pretrain_manager.model,
                                                  os.path.join(args.model_output_path, 'pretrain'), self.device)

            # 模型训练
        if args.train:
            # 计算训练轮数
            args.num_train_epochs = (1 - args.thres) / args.delta
            # 设置优化器和学习率调度器
            self.optimizer, self.scheduler = set_optimizer(args, self.model, args.lr)

            if args.freeze_train_bert_parameters:
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                self.model = freeze_bert_parameters(self.model, args.multimodal_method)

            # 加载预训练模型
            self.load_pretrained_model(self.pretrained_model)
        else:
            # 如果不进行训练，恢复模型
            self.model = restore_model(self.model, args.model_output_path, self.device)

        self.efficiency_recorder.record_parameters(
            self.model,
            stage='main_train_ready' if args.train else 'inference_ready',
            paper_snapshot=True,
        )

    def clustering(self, args, init='k-means++', threshold=0.25):
        # 聚类方法，传入参数 args、初始化方法 init 和阈值 threshold

        # 获取训练数据的输出，返回特征
        outputs = self._get_outputs(args, mode='train', return_feats=True)
        # 获取特征
        feats = outputs['feats']
        # 获取真实标签
        y_true = outputs['y_true']

        if init == 'k-means++':
            # 如果初始化方法为 k-means++

            self.logger.info('Initializing centroids with K-means++...')
            # 记录开始时间
            start = time.time()
            # 使用 KMeans++ 初始化聚类中心并拟合数据
            km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed, init='k-means++').fit(feats)

            # 获取聚类中心和分配的标签
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            # 记录结束时间
            end = time.time()
            self.logger.info('K-means++ used %s s', round(end - start, 2))

        elif init == 'centers':
            # 如果初始化方法为 centers

            start = time.time()
            # 使用指定的聚类中心初始化并拟合数据
            km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed, init=self.centroids).fit(feats)
            # 获取聚类中心和分配的标签
            km_centroids, assign_labels = km.cluster_centers_, km.labels_
            end = time.time()
            self.logger.info('K-means used %s s', round(end - start, 2))

        # 更新聚类中心
        self.centroids = km_centroids

        # 初始化选择的索引列表
        select_ids = []

        # 遍历每个聚类
        for cluster_id in range(self.num_labels):
            # 获取当前聚类的样本特征
            cluster_samples = feats[assign_labels == cluster_id]
            # 获取当前聚类的样本索引
            pos = list(np.where(assign_labels == cluster_id)[0])

            cutoff = max(int(len(cluster_samples) * threshold), 1)
            k_candidate_proportions = np.arange(0.1, 0.32, 0.02).tolist()

            if cutoff == 1:
                select_ids.extend(pos)

            else:

                best_sorted_indices = None
                best_eval_score = 0
                best_k_cand = None

                for k_cand in k_candidate_proportions:

                    k = max(int(len(cluster_samples) * k_cand), 1)
                    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(cluster_samples)
                    distances, indices = nbrs.kneighbors(cluster_samples)

                    reachable_distances = np.mean(distances[:, 1:], axis=1)
                    density = 1 / reachable_distances
                    sorted_indices = np.argsort(density)

                    tmp_select_indices = sorted_indices[-cutoff:]
                    tmp_select_pos = [pos[i] for i in tmp_select_indices]

                    tmp_feats = feats[tmp_select_pos]
                    tmp_assign_labels = assign_labels[tmp_select_pos]

                    tree = KDTree(tmp_feats)
                    _, ind = tree.query(tmp_feats, k=2)
                    nearest_neighbor_distances = np.array(
                        [tmp_feats[i] - tmp_feats[ind[i, 1]] for i in range(len(tmp_feats))])
                    tmp_eval_score = np.mean(np.linalg.norm(nearest_neighbor_distances, axis=1))

                    if tmp_eval_score > best_eval_score:
                        best_eval_score = tmp_eval_score
                        best_k_cand = k_cand
                        best_sorted_indices = sorted_indices

                select_indices = best_sorted_indices[-cutoff:]
                select_pos = [pos[i] for i in select_indices]
                select_ids.extend(select_pos)

        return np.array(assign_labels), select_ids, feats

    # 训练方法
    # 训练方法
    # 训练方法
    def _train(self, args):

        # 将模型移动到指定设备
        self.model.to(self.device)

        # 遍历训练轮数，显示进度条
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):

            efficiency_segment = self.efficiency_recorder.begin_segment(
                'main_train', 'train', epoch_stage=epoch + 1
            )

            # 计算当前轮次的阈值
            threshold = args.thres + args.delta * epoch

            # 根据轮次选择初始化机制
            init_mechanism = 'k-means++' if epoch == 0 else 'centers'
            # 进行聚类
            # 进行聚类
            # 进行聚类，获取伪标签、选择的索引和特征
            pseudo_labels, select_ids, feats = self.clustering(args, init=init_mechanism, threshold=threshold)

            # 记录日志
            if epoch > 0:
                self.logger.info("***** Epoch: %s *****", str(epoch))
                self.logger.info('Supervised Training Loss: %f', np.round(tr_sup_loss, 5))
                self.logger.info('Alignment Loss: %f', np.round(tr_align_loss, 5))
                if len(non_select_ids) != 0:
                    self.logger.info('Unsupervised Training Loss: %f', np.round(tr_unsup_loss, 5))

            # 将伪标签设置为训练输出的标签
            self.train_outputs['label_ids'] = pseudo_labels
            # 将选择的索引设置为训练输出的选择索引
            self.train_outputs['select_ids'] = select_ids

            # 获取伪监督训练数据加载器
            _, pseudo_sup_train_dataloader = get_pseudo_dataloader(args=args, \
                                                                   train_outputs=self.train_outputs, mode='pretrain')

            # 设置模型为训练模式
            self.model.train()
            # 初始化监督训练损失
            tr_sup_loss = 0
            tr_align_loss = 0
            sup_contrast_weighted_sum = 0.0
            sup_align_weighted_sum = 0.0
            sup_total_weighted_sum = 0.0
            sup_samples_seen = 0
            # 初始化训练样本数和训练步数
            nb_tr_examples, nb_tr_steps = 0, 0

            # 遍历伪监督训练数据加载器，显示进度条
            for batch_sup in tqdm(pseudo_sup_train_dataloader, desc='Iteration'):

                # 获取文本特征、视频特征、音频特征，并移动到指定设备
                text_feats = batch_sup['text_feats'].to(self.device)
                video_feats = batch_sup['video_feats'].to(self.device)
                audio_feats = batch_sup['audio_feats'].to(self.device)

                # 获取标签 ID 并移动到指定设备
                label_ids = batch_sup['label_ids'].to(self.device)

                # 启用梯度计算
                with torch.set_grad_enabled(True):

                    # 获取模型输出
                    _, mlp_output_a = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats,
                                                 mode='train-mm')
                    _, mlp_output_b = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device),
                                                 mode='train-mm')
                    _, mlp_output_c = self.model(text_feats, video_feats, audio_feats, mode='train-mm')

                    # 对模型输出进行归一化
                    norm_mlp_output_a = F.normalize(mlp_output_a)
                    norm_mlp_output_b = F.normalize(mlp_output_b)
                    norm_mlp_output_c = F.normalize(mlp_output_c)

                    # 拼接对比逻辑值
                    contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1),
                                                    norm_mlp_output_c.unsqueeze(1)), dim=1)
                    # 计算监督对比损失
                    loss_sup = self.contrast_criterion(contrastive_logits, labels=label_ids,
                                                       temperature=args.train_temperature_sup, device=self.device)

                    # 计算多模态对齐正则化损失 (使用InfoNCE损失)
                    batch_size = norm_mlp_output_a.size(0)
                    temperature = args.align_temperature

                    # 计算模态a和b之间的InfoNCE损失
                    sim_a_b = torch.exp(torch.mm(norm_mlp_output_a, norm_mlp_output_b.t()) / temperature)
                    pos_sim_a_b = torch.diag(sim_a_b)
                    align_loss_a_b = -torch.log(pos_sim_a_b / sim_a_b.sum(dim=1)).mean()

                    # 计算模态a和c之间的InfoNCE损失
                    sim_a_c = torch.exp(torch.mm(norm_mlp_output_a, norm_mlp_output_c.t()) / temperature)
                    pos_sim_a_c = torch.diag(sim_a_c)
                    align_loss_a_c = -torch.log(pos_sim_a_c / sim_a_c.sum(dim=1)).mean()

                    # 计算模态b和c之间的InfoNCE损失
                    sim_b_c = torch.exp(torch.mm(norm_mlp_output_b, norm_mlp_output_c.t()) / temperature)
                    pos_sim_b_c = torch.diag(sim_b_c)
                    align_loss_b_c = -torch.log(pos_sim_b_c / sim_b_c.sum(dim=1)).mean()

                    # 计算总的对齐损失
                    align_loss = (align_loss_a_b + align_loss_a_c + align_loss_b_c) / 3.0

                    # 设置损失为监督对比损失加上多模态对齐损失
                    # 使用权重系数控制对齐损失的影响
                    # 多模态对齐正则化的目的是让不同模态（text/video/audio）在共享空间中更紧密对齐，提升表示一致性
                    # 这里使用InfoNCE损失来计算不同模态之间的对齐程度，温度系数控制分布的平滑度
                    loss = loss_sup + args.align_weight * align_loss

                    # 清空梯度
                    self.optimizer.zero_grad()
                    # 反向传播
                    # 反向传播
                    # 反向传播
                    loss.backward()

                    if args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_(
                            [param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                    # 累加监督训练损失和对齐损失
                    loss_sup_value = loss_sup.item()
                    align_loss_value = align_loss.item()
                    total_loss_value = loss.item()
                    batch_size = int(label_ids.size(0))
                    tr_sup_loss += loss_sup_value
                    tr_align_loss += align_loss_value
                    sup_contrast_weighted_sum += loss_sup_value * batch_size
                    sup_align_weighted_sum += align_loss_value * batch_size
                    sup_total_weighted_sum += total_loss_value * batch_size
                    sup_samples_seen += batch_size
                    # 增加训练步数
                    nb_tr_steps += 1

                    # 更新模型参数
                    self.optimizer.step()
                    # 更新学习率调度器
                    self.scheduler.step()

                    # 清空缓存
                    torch.cuda.empty_cache()

            # 计算平均监督训练损失和对齐损失
            tr_sup_loss /= nb_tr_steps
            tr_align_loss /= nb_tr_steps
            sup_optimizer_steps = nb_tr_steps

            # 获取未选择的索引--低质量样本
            # 获取未选择的索引--低质量样本
            # 获取未选择的索引--低质量样本
            non_select_ids = [i for i in range(len(pseudo_labels)) if i not in select_ids]
            unsup_pseudo_labels = pseudo_labels[non_select_ids]
            tr_unsup_loss = None
            unsup_weighted_loss_sum = 0.0
            unsup_samples_seen = 0
            unsup_optimizer_steps = 0

            # 对低质量样本进行处理
            if len(non_select_ids) != 0:

                self.train_outputs['select_ids'] = non_select_ids
                _, pseudo_unsup_train_dataloader = get_pseudo_dataloader(args=args, \
                                                                         train_outputs=self.train_outputs,
                                                                         mode='pretrain')

                tr_unsup_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0

                for batch_unsup in tqdm(pseudo_unsup_train_dataloader, desc='Iteration'):

                    unsup_text_feats = batch_unsup['text_feats'].to(self.device)
                    unsup_video_feats = batch_unsup['video_feats'].to(self.device)
                    unsup_audio_feats = batch_unsup['audio_feats'].to(self.device)

                    with torch.set_grad_enabled(True):

                        _, mlp_output_a = self.model(unsup_text_feats,
                                                     torch.zeros_like(unsup_video_feats).to(self.device),
                                                     unsup_audio_feats, mode='train-mm')
                        _, mlp_output_b = self.model(unsup_text_feats, unsup_video_feats,
                                                     torch.zeros_like(unsup_audio_feats).to(self.device),
                                                     mode='train-mm')
                        _, mlp_output_c = self.model(unsup_text_feats, unsup_video_feats, unsup_audio_feats,
                                                     mode='train-mm')

                        norm_mlp_output_a = F.normalize(mlp_output_a)
                        norm_mlp_output_b = F.normalize(mlp_output_b)
                        norm_mlp_output_c = F.normalize(mlp_output_c)

                        contrastive_logits = torch.cat((norm_mlp_output_a.unsqueeze(1), norm_mlp_output_b.unsqueeze(1),
                                                        norm_mlp_output_c.unsqueeze(1)), dim=1)
                        loss_unsup = self.contrast_criterion(contrastive_logits,
                                                             temperature=args.train_temperature_unsup,
                                                             device=self.device)

                        loss = loss_unsup

                        self.optimizer.zero_grad()
                        loss.backward()

                        if args.grad_clip != -1.0:
                            torch.nn.utils.clip_grad_value_(
                                [param for param in self.model.parameters() if param.requires_grad], args.grad_clip)

                        loss_value = loss.item()
                        batch_size = int(unsup_text_feats.size(0))
                        tr_unsup_loss += loss_value
                        unsup_weighted_loss_sum += loss_value * batch_size
                        unsup_samples_seen += batch_size
                        nb_tr_steps += 1

                        self.optimizer.step()
                        self.scheduler.step()

                        torch.cuda.empty_cache()

                tr_unsup_loss /= nb_tr_steps
                unsup_optimizer_steps = nb_tr_steps

            self.efficiency_recorder.end_segment(efficiency_segment)
            total_samples_seen = sup_samples_seen + unsup_samples_seen
            self.efficiency_recorder.record_epoch(
                stage='main_train',
                epoch_stage=epoch + 1,
                segment=efficiency_segment,
                loss_components={
                    'supervised_contrastive': (
                        sup_contrast_weighted_sum / sup_samples_seen
                        if sup_samples_seen else None
                    ),
                    'icla_alignment': (
                        sup_align_weighted_sum / sup_samples_seen
                        if sup_samples_seen else None
                    ),
                    'unsupervised_contrastive': (
                        unsup_weighted_loss_sum / unsup_samples_seen
                        if unsup_samples_seen else None
                    ),
                },
                samples_seen=total_samples_seen,
                optimizer_steps=sup_optimizer_steps + unsup_optimizer_steps,
                learning_rates=[group['lr'] for group in self.optimizer.param_groups],
                loss_total_mean=(
                    (sup_total_weighted_sum + unsup_weighted_loss_sum)
                    / total_samples_seen if total_samples_seen else None
                ),
            )

        # 如果需要保存模型，则保存模型
        if args.save_model:
            with self.efficiency_recorder.segment('main_checkpoint', 'checkpoint'):
                save_model(self.model, args.model_output_path)

    # 测试方法
    # 测试方法
    # 测试方法
    def _test(self, args):

        inference_segment = self.efficiency_recorder.begin_segment(
            'final_test', 'inference', details={'split': 'test'}
        )
        self.model.to(self.device)

        # 获取测试数据的输出，返回特征
        outputs = self._get_outputs(args, mode='test', return_feats=True)
        # 获取特征
        feats = outputs['feats']
        # 获取真实标签
        y_true = outputs['y_true']

        # 使用 KMeans++ 初始化聚类中心并拟合数据
        km = KMeans(n_clusters=self.num_labels, n_jobs=-1, random_state=args.seed, \
                    init='k-means++').fit(feats)

        # 获取预测标签
        y_pred = km.labels_

        # 计算聚类评估指标
        test_results = clustering_score(y_true, y_pred)
        self.efficiency_recorder.end_segment(inference_segment)
        self.efficiency_recorder.record_evaluation(
            stage='final_test',
            segment=inference_segment,
            metrics=test_results,
            split='test',
            n_samples=len(y_true),
            role='final',
        )

        # 日志，输出每个评估指标的值
        self.logger.info("***** Test results *****")
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        # 将真实标签添加到测试结果
        test_results['y_true'] = y_true
        # 将预测标签添加到测试结果
        test_results['y_pred'] = y_pred
        # 同步返回嵌入，便于 save_results 保存 feats.npy
        test_results['feats'] = feats

        return test_results

    # 获取输出方法，传入参数 args、模式 mode、是否返回特征 return_feats 和模态 modality
    # 输出的格式及特征保存的格式
    def _get_outputs(self, args, mode, return_feats=False, modality='tva'):

        if mode == 'test':

            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Get Outputs"):

            text_feats = batch['text_feats'].to(self.device)
            video_feats = batch['video_feats'].to(self.device)
            audio_feats = batch['audio_feats'].to(self.device)
            label_ids = batch['label_ids'].to(self.device)

            with torch.set_grad_enabled(False):
                if modality == 'tva':
                    features, _ = self.model(text_feats, video_feats, audio_feats, mode='train-mm')
                elif modality == 't0a':
                    features, _ = self.model(text_feats, torch.zeros_like(video_feats).to(self.device), audio_feats,
                                             mode='train-mm')
                elif modality == 'tv0':
                    features, _ = self.model(text_feats, video_feats, torch.zeros_like(audio_feats).to(self.device),
                                             mode='train-mm')

                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, features))

        # 将特征移动到 CPU 并转换为 NumPy 数组
        feats = total_features.cpu().numpy()
        # 将标签移动到 CPU 并转换为 NumPy 数组
        y_true = total_labels.cpu().numpy()

        if return_feats:
            outputs = {
                'y_true': y_true,  # 真实标签
                'feats': feats  # 特征
            }

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim=1)

            # 将预测标签移动到 CPU 并转换为 NumPy 数组
            y_pred = total_preds.cpu().numpy()
            # 将逻辑值移动到 CPU 并转换为 NumPy 数组
            y_logits = total_logits.cpu().numpy()

            outputs = {
                'y_true': y_true,  # 真实标签
                'y_pred': y_pred,  # 预测标签
                'logits': y_logits,  # 逻辑值
                'feats': feats  # 特征
            }

        return outputs

    # 加载预训练模型方法，传入预训练模型
    def load_pretrained_model(self, pretrained_model):

        # 获取预训练模型的状态字典
        pretrained_dict = pretrained_model.state_dict()
        # 定义 MLP 参数键列表
        mlp_params = ['method_model.mlp_head_train.2.weight', 'method_model.mlp_head_train.2.bias',
                      'method_model.classifier.weight', 'method_model.classifier.bias']

        # 过滤掉 MLP 参数
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in mlp_params}
        # 加载预训练模型的状态字典到当前模型，strict 设置为 False 允许不匹配的键
        self.model.load_state_dict(pretrained_dict, strict=False)
