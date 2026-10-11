import os
import pandas as pd
import torch.utils.data as util_data
from sklearn import preprocessing
from torch.utils.data import Dataset
import numpy as np
import torch

class VirtualAugSamples(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}

    
class ExplitAugSamples(Dataset):
    def __init__(self, train_x, train_x1, train_x2, train_y):
        assert len(train_y) == len(train_x) == len(train_x1) == len(train_x2)
        self.train_x = train_x
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.train_y = train_y
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'augmentation_1': self.train_x1[idx], 'augmentation_2': self.train_x2[idx], 'label': self.train_y[idx]}
       

def explict_augmentation_loader(args, datapath, dataname):
    train_data = pd.read_csv(os.path.join(datapath, dataname+".csv"))
    train_text = train_data[args.text_column].fillna('.').values
    train_text1 = train_data[args.augmentation_1_column].fillna('.').values
    train_text2 = train_data[args.augmentation_2_column].fillna('.').values
    train_label = train_data[args.label_column].astype(int).values

    train_dataset = ExplitAugSamples(train_text, train_text1, train_text2, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader


def virtual_augmentation_loader(args, datapath, dataname):
    train_data = pd.read_csv(os.path.join(datapath, dataname+".csv"))
    train_text = train_data[args.text_column].fillna('.').values
    train_label = train_data[args.label_column].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)  
    return train_loader


def unshuffle_loader(args, datapath, dataname_test):
    train_data = pd.read_csv(os.path.join(datapath, dataname_test+".csv"))
    train_text = train_data[args.text_column].fillna('.').values
    train_label = train_data[args.label_column].astype(int).values

    train_dataset = VirtualAugSamples(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)  
    return train_loader


class AugmentedMMDataset(Dataset):
    def __init__(self, label_ids, text_data, video_data, audio_data, tokenizer=None, args=None):
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.tokenizer = tokenizer
        self.args = args
        
    def __len__(self):
        return len(self.label_ids)
    
    def __getitem__(self, idx):
        # 基本数据
        label_id = self.label_ids[idx]
        text_feat = self.text_data[idx]
        video_feat = self.video_data['feats'][idx]
        audio_feat = self.audio_data['feats'][idx]
        
        # 如果有tokenizer，创建两个文本增强视图
        if self.tokenizer is not None and self.args is not None:
            # 获取原始输入ID和掩码
            input_ids = torch.tensor(text_feat[0])
            attention_mask = torch.tensor(text_feat[1])
            segment_ids = torch.tensor(text_feat[2])
            
            # 创建两个增强视图
            from methods.unsupervised.UMC_ICLA.utils import view_generator
            generator = view_generator(self.tokenizer, self.args)
            input_ids_1, attention_mask_1 = generator.random_token_erase(input_ids, attention_mask)
            input_ids_2, attention_mask_2 = generator.random_token_erase(input_ids, attention_mask)
            
            # 组合增强视图
            text_feat_aug = [
                [input_ids_1.tolist(), attention_mask_1.tolist(), segment_ids.tolist()],
                [input_ids_2.tolist(), attention_mask_2.tolist(), segment_ids.tolist()]
            ]
            
            return {
                'label_ids': label_id,
                'text_feats': text_feat,
                'text_feats_aug': text_feat_aug,
                'video_feats': video_feat,
                'audio_feats': audio_feat
            }
        
        return {
            'label_ids': label_id,
            'text_feats': text_feat,
            'video_feats': video_feat,
            'audio_feats': audio_feat
        }


def get_augmented_dataloader(args, data, tokenizer=None):
    """
    创建带有数据增强的数据加载器
    """
    train_dataset = AugmentedMMDataset(
        data['train']['label_ids'],
        data['train']['text_data'],
        data['train']['video_data'],
        data['train']['audio_data'],
        tokenizer,
        args
    )
    
    train_dataloader = util_data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers if hasattr(args, 'num_workers') else 4
    )
    
    return train_dataloader
