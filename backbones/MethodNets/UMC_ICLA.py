import torch.nn as nn
import torch.nn.functional as F
import torch

class UMCICLAModel(nn.Module):

    def __init__(self, args, backbone):

        super(UMCICLAModel, self).__init__()
        self.backbone = backbone
        activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        args.feat_dim = args.base_dim
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.mlp_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(args.base_dim, args.base_dim), 
        )
        
        # 添加对比学习头
        self.contrast_head = nn.Sequential(
            nn.Linear(args.base_dim, args.base_dim),
            nn.ReLU(),
            nn.Linear(args.base_dim, args.base_dim)
        )
        

    def forward(self, text, video, audio, mode='train'):
        
        if mode == 'pretrain-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.mlp_head(features)
            return mlp_output

        elif mode == 'train-mm':
            features = self.backbone(text, video, audio, mode='features')
            mlp_output = self.mlp_head(features)

            return features, mlp_output
        
        elif mode == 'virtual':
            # 处理虚拟增强数据
            # 假设text包含两个增强视图的输入
            input_ids, attention_mask = torch.unbind(text[0], dim=1), torch.unbind(text[1], dim=1)
            
            # 获取两个视图的特征
            embd1 = self.get_text_embeddings(input_ids[0], attention_mask[0])
            embd2 = self.get_text_embeddings(input_ids[1], attention_mask[1])
            
            return embd1, embd2
                
    def get_text_embeddings(self, input_ids, attention_mask):
        """获取文本嵌入"""
        # 使用backbone中的文本嵌入器
        text_embedding = self.backbone.text_embedding([input_ids, attention_mask, torch.zeros_like(input_ids)])
        text_embedding = text_embedding[:, 0]  # 取CLS token
        text_embedding = self.backbone.text_layer(text_embedding)
        return text_embedding
    
    def contrast_logits(self, embd1, embd2=None):
        """计算对比逻辑"""
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 is not None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1
    
    def get_cluster_prob(self, embeddings):
        """获取聚类概率"""
        return F.softmax(self.contrast_head(embeddings), dim=1)
                
