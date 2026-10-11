import os
import logging
import csv
import copy

from .mm_pre import MMDataset
from .text_pre import get_t_data
from .utils import get_v_a_data
from .__init__ import benchmarks

__all__ = ['DataManager']

# 数据管理器类，用于管理多模态数据的加载和处理
class DataManager:

    # 构造函数，传入参数 args
    def __init__(self, args):
        
        self.logger = logging.getLogger(args.logger_name)  # 获取日志记录器
        bm = benchmarks[args.dataset]  # 获取 benchmarks 字典中对应数据集的配置
        max_seq_lengths, feat_dims = bm['max_seq_lengths'], bm['feat_dims']  # 获取最大序列长度和特征维度信息

        # 将文本、视频、音频的最大序列长度和特征维度分别赋值给 args 对应的属性
        args.text_seq_len, args.video_seq_len, args.audio_seq_len = max_seq_lengths['text'], max_seq_lengths['video'], max_seq_lengths['audio']
        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim = feat_dims['text'], feat_dims['video'], feat_dims['audio']

        # 调用 get_data 函数，传入 args 和日志记录器，获取多模态数据和训练输出
        self.mm_data, self.train_outputs = get_data(args, self.logger) 
        
def get_data(args, logger):

    # 构造数据路径，由 args.data_path 和 args.dataset 拼接而成
    data_path = os.path.join(args.data_path, args.dataset)
    bm = benchmarks[args.dataset]  # 获取 benchmarks 字典中对应数据集的配置
    
    label_list = copy.deepcopy(bm["labels"])  # 拷贝标签列表，避免修改原数据
    logger.info('Lists of intent labels are: %s', str(label_list))  # 记录日志，输出标签列表
      
    args.num_labels = len(label_list)  # 设置 args.num_labels 为标签数量
    
    logger.info('data preparation...')  # 记录日志，表示开始准备数据

    # 获取训练数据的索引和标签 ID
    train_data_index, train_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'train.tsv'))
    # 获取开发集数据的索引和标签 ID
    dev_data_index, dev_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'dev.tsv'))

    # 将训练集和开发集的数据索引和标签 ID 合并
    train_data_index = train_data_index + dev_data_index
    train_label_ids = train_label_ids + dev_label_ids

    # 获取测试数据的索引和标签 ID
    test_data_index, test_label_ids = get_indexes_annotations(args, bm, label_list, os.path.join(data_path, 'test.tsv'))
    args.num_train_examples = len(train_data_index)  # 设置 args.num_train_examples 为训练样本数量

    # 构造包含数据路径和索引的字典
    data_args = {
        'data_path': data_path,
        'train_data_index': train_data_index,
        'test_data_index': test_data_index,
    }

    # 获取文本数据
    text_data = get_t_data(args, data_args)

    # 构造视频特征路径
    video_feats_path = os.path.join(data_args['data_path'], args.video_data_path, args.video_feats_path)
    # 获取视频数据
    video_data = get_v_a_data(data_args, video_feats_path, args.video_seq_len)

    # 构造音频特征路径
    audio_feats_path = os.path.join(data_args['data_path'], args.audio_data_path, args.audio_feats_path)
    # 获取音频数据
    audio_data = get_v_a_data(data_args, audio_feats_path, args.audio_seq_len)

    # 构造多模态训练数据和测试数据
    mm_train_data = MMDataset(train_label_ids, text_data['train'], video_data['train'], audio_data['train'])
    mm_test_data = MMDataset(test_label_ids, text_data['test'], video_data['test'], audio_data['test'])

    # 将多模态数据存入字典
    mm_data = {'train': mm_train_data, 'test': mm_test_data}

    # 构造训练输出字典，包含文本、视频、音频数据和标签 ID
    train_outputs = {
        'text': text_data['train'],
        'video': video_data['train'],
        'audio': audio_data['train'],
        'label_ids': train_label_ids,
    }

    # 返回多模态数据和训练输出
    return mm_data, train_outputs

# 获取索引和注释的函数（三个数据集稍有区别），返回 索引列表 和 标签ID列表
def get_indexes_annotations(args, bm, label_list, read_file_path):

    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    with open(read_file_path, 'r') as f:

        data = csv.reader(f, delimiter="\t")
        indexes = []
        label_ids = []

        for i, line in enumerate(data):
            if i == 0:
                continue
            
            if args.dataset in ['MIntRec']:
                index = '_'.join([line[0], line[1], line[2]])  # 构造索引，由前三列拼接而成
                indexes.append(index)
                
                label_id = label_map[line[4]]  # 获取标签 ID，由第 5 列对应的标签映射得到
            
            elif args.dataset in ['MELD-DA']:
                label_id = label_map[line[3]]  # 获取标签 ID，由第 4 列对应的标签映射得到
                
                index = '_'.join([line[0], line[1]])  # 构造索引，由前两列拼接而成
                indexes.append(index)
            
            elif args.dataset in ['IEMOCAP-DA']:
                label_id = label_map[line[2]]  # 获取标签 ID，由第 3 列对应的标签映射得到
                index = line[0]  # 构造索引，由第 1 列得到
                indexes.append(index)
            
            label_ids.append(label_id)
    
    return indexes, label_ids