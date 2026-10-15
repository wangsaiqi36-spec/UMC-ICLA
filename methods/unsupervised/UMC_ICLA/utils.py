import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
from data.mm_pre import MMDataset
import numpy as np
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from data.augmentation import AugmentedMMDataset

def get_pseudo_dataloader(args, train_outputs, mode='pretrain', pseudo_labels=None):
    # 获取伪数据加载器函数
    '''
    video_data: {'feats': xxx, 'lengths': xxx}
    audio_data: {'feats': xxx, 'lengths': xxx}
    text_data: [input_ids, input_mask, segment_ids]
    '''
    text = train_outputs['text']  # 获取文本数据
    video = train_outputs['video']  # 获取视频数据
    audio = train_outputs['audio']  # 获取音频数据

    # 获取选择的索引
    if 'select_ids' in train_outputs.keys():
        select_ids = train_outputs['select_ids']

    # 获取伪标签
    if pseudo_labels is None:
        pseudo_labels = train_outputs['label_ids']

    # 如果有选择的索引
    if 'select_ids' in train_outputs.keys():

        # 根据选择的索引筛选伪标签
        pseudo_labels = pseudo_labels[select_ids]

        # 根据选择的索引筛选文本数据
        text = [text[i] for i in select_ids]

        new_video = {}
        # 根据选择的索引筛选视频长度数据
        new_video['lengths'] = [video['lengths'][i] for i in select_ids]
        # 根据选择的索引筛选视频特征数据
        new_video['feats'] = [video['feats'][i] for i in select_ids]

        new_audio = {}
        # 根据选择的索引筛选音频长度数据
        new_audio['lengths'] = [audio['lengths'][i] for i in select_ids]
        # 根据选择的索引筛选音频特征数据
        new_audio['feats'] = [audio['feats'][i] for i in select_ids]

        # 将伪标签转换为张量并增加维度
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        # 创建新的多模态数据集
        train_data = MMDataset(train_label_ids, text, new_video, new_audio)

    else:
        # 将伪标签转换为张量并增加维度
        train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
        # 创建多模态数据集
        train_data = MMDataset(train_label_ids, text, video, audio)

    # 创建随机采样器
    sampler = RandomSampler(train_data)

    # 如果是预训练模式
    if mode == 'pretrain':
        # 创建数据加载器，使用 预训练 批次大小
        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.pretrain_batch_size)
    else:
        # 创建数据加载器，使用 训练 批次大小
        train_dataloader = DataLoader(train_data, sampler = sampler, batch_size = args.train_batch_size)

    # 返回训练数据和数据加载器
    return train_data, train_dataloader

def get_augmented_dataloader(args, train_outputs, mode='pretrain', pseudo_labels=None):
    # 获取虚拟增强数据加载器函数
    '''
    为 UMC-ICLA 模型提供数据增强功能
    '''
    text = train_outputs['text']  # 获取文本数据
    video = train_outputs['video']  # 获取视频数据
    audio = train_outputs['audio']  # 获取音频数据

    # 获取伪标签
    if pseudo_labels is None:
        pseudo_labels = train_outputs['label_ids']
    
    # 将伪标签转换为张量并增加维度
    train_label_ids = torch.tensor(pseudo_labels).unsqueeze(1)
    
    # 创建增强的多模态数据集
    train_data = AugmentedMMDataset(train_label_ids, text, video, audio, args)

    # 创建随机采样器
    sampler = RandomSampler(train_data)

    # 根据模式选择批次大小
    if mode == 'pretrain':
        # 创建数据加载器，使用预训练批次大小
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=args.pretrain_batch_size)
    else:
        # 创建数据加载器，使用训练批次大小
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=args.train_batch_size)

    # 返回训练数据和数据加载器
    return train_data, train_dataloader

class view_generator:
    # 视图生成器类
    
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer  # 保存分词器
        self.args = args  # 保存参数

    # 随机令牌擦除方法
    def random_token_erase(self, input_x, input_mask, max_seq_length=30, mode = 'text'):
        
        aug_input_x = []  # 初始化增强后的输入
        aug_input_mask = []  # 初始化增强后的掩码

        # 遍历输入和掩码
        for inp_x, inp_m in zip(input_x, input_mask):
            # 如果是文本模式
            if mode == 'text':
                # 获取特殊令牌掩码
                special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp_x, already_has_special_tokens=True)
                # 获取句子令牌索引
                sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]

                # 创建索引数组
                inds = np.arange(len(sent_tokens_inds))
                # 随机选择要擦除的索引
                masked_inds = np.random.choice(inds, size = int(len(inds) * self.args.re_prob), replace = False)
                # 获取句子中要擦除的索引
                sent_masked_inds = sent_tokens_inds[masked_inds]

                # 删除擦除位置的令牌
                inp_x = np.delete(inp_x, sent_masked_inds)
                # 填充到最大序列长度
                inp_x = F.pad(inp_x, (0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)

                # 删除擦除位置的掩码
                inp_m = np.delete(inp_m, sent_masked_inds)
                # 填充掩码到最大序列长度
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)
            
            else:
                # 获取句子令牌索引
                sent_tokens_inds = np.where(inp_m.numpy() == 1)[0]

                # 随机选择擦除起始索引
                erase_start_ind = np.random.choice(sent_tokens_inds)
                # 计算擦除结束索引
                erase_end_ind = min(erase_start_ind + max(1, int(self.args.re_prob * (len(sent_tokens_inds) - erase_start_ind))), len(sent_tokens_inds))
                # 获取擦除索引范围
                erase_inds = sent_tokens_inds[erase_start_ind: erase_end_ind]

                # 删除擦除位置的令牌
                inp_x = np.delete(inp_x, erase_inds, axis = 0)
                # 转换为张量
                inp_x = torch.from_numpy(inp_x)
                # 填充到最大序列长度
                inp_x = F.pad(inp_x, (0, 0, 0, max_seq_length - len(inp_x)), mode = 'constant', value = 0)

                # 删除擦除位置的掩码
                inp_m = np.delete(inp_m, erase_inds)
                # 填充掩码到最大序列长度
                inp_m = F.pad(inp_m, (0, max_seq_length - len(inp_m)), 'constant', 0)

            aug_input_x.append(inp_x)
            aug_input_mask.append(inp_m)

        # 将列表转换为张量
        aug_input_x = torch.stack(aug_input_x, dim = 0)
        aug_input_mask = torch.stack(aug_input_mask, dim = 0)

        # 返回增强后的输入和掩码
        return aug_input_x, aug_input_mask

        
# 设置优化器函数
def set_optimizer(args, model, lr, mode='train'):

    num_train_epochs = args.num_train_epochs  # 获取训练轮数

    param_optimizer = list(model.named_parameters())  # 获取模型的命名参数列表
    # 定义不进行衰减的参数名称
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # 对不需要衰减的参数设置权重衰减
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        # 对需要衰减的参数设置权重衰减为 0
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # 创建 AdamW 优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr = lr, correct_bias=False)

    # 计算训练优化步骤数
    num_train_optimization_steps = int(args.num_train_examples / args.train_batch_size) * num_train_epochs
    # 计算 warmup 步骤数
    num_warmup_steps = int(args.num_train_examples * num_train_epochs * args.warmup_proportion / args.train_batch_size)

    # 创建学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    # 返回优化器和调度器
    return optimizer, scheduler
