import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from jack_transformer_share import MyTransformer
import torch.nn.functional as F
import math

# let's define a simple model that can deal with multimodal variable length sequence
class AOTN(nn.Module):
    def __init__(self, config):
        super(AOTN, self).__init__()
        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()

        ##########################################
        # 1. 模态特征提取
        ##########################################
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        ##########################################
        # 2. 模态对齐
        ##########################################

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', 
                                  nn.Linear(in_features=768, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        self.batchnorm = nn.BatchNorm1d(2, affine=False)

        ##########################################
        # 3. 模态不变和模态特定
        ##########################################
        # 模态特定
        # self.tf_encoder_private_a = MyTransformer(d_emb_1=4*input_sizes[2], d_emb_2=4*input_sizes[2], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # self.tf_encoder_private_v = MyTransformer(d_emb_1=4*input_sizes[1], d_emb_2=4*input_sizes[1], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # self.tf_encoder_private_t = MyTransformer(d_emb_1=768, d_emb_2=768, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        self.tf_encoder_private_a = MyTransformer(d_emb_1=4*input_sizes[2], d_emb_2=4*input_sizes[1], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        self.tf_encoder_private_v = MyTransformer(d_emb_1=4*input_sizes[1], d_emb_2=768, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        self.tf_encoder_private_t = MyTransformer(d_emb_1=768, d_emb_2=4*input_sizes[2], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # 模态不变
        self.tf_encoder_share = MyTransformer(d_emb_1=self.config.hidden_size, d_emb_2=self.config.hidden_size, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # 模态对齐
        self.fc_a = nn.Linear(4*input_sizes[2], self.config.hidden_size)
        self.fc_v = nn.Linear(4*input_sizes[1], self.config.hidden_size)
        self.fc_t = nn.Linear(768, self.config.hidden_size)
        
        ##########################################
        # 4. 模态融合
        ##########################################
        self.fusion3 = nn.Sequential(
            nn.Conv1d(self.config.hidden_size, self.config.hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),            
            nn.Conv1d(self.config.hidden_size//2, self.config.hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),            
            nn.Conv1d(self.config.hidden_size//2, 1, kernel_size=3, stride=1, padding=1),            
            nn.AdaptiveAvgPool1d(1)
            # nn.AdaptiveMaxPool1d(1)
        )
        
        # 额外引入一层通信层用来提升性能
        self.MLP_Communicator1 = MLP_Communicator(self.config.hidden_size, 2, hidden_size=64, depth=1)
        self.MLP_Communicator2 = MLP_Communicator(self.config.hidden_size, 2, hidden_size=64, depth=1)

        # 使用可学习的 cls 和 positionencoing 避免模型过拟合
        self.cls_token = nn.Parameter(torch.zeros(self.config.batch_size, 1, self.config.hidden_size)) # (32,1,256)
        self.pos_embed = nn.Parameter(torch.zeros(self.config.batch_size, 12 + 2, self.config.hidden_size)) # (32,14,256) 12 就是6种模态的融合长度 +2 就是表示将模态不变和模态特定区分开

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm): # sequence.shape = torch.Size([59, 32, 35])
        packed_sequence = pack_padded_sequence(sequence, lengths) # packed_sequence: ([645, 35]) & ([59]) % None & None

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)  # packed_h1: ([645, 70]) & ([59]) & None & None /////// final_h1.shape = ([2, 32, 35])
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1) # padded_h1.shape = ([59, 32, 70])
        normed_h1 = layer_norm(padded_h1) # normed_h1.shape = ([59, 32, 70])
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths) # packed_normed_h1: ([645, 70]) & ([59]) & None & None 

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1) # final_h2.shape = ([2, 32, 35])
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2 # final_h1.shape = ([2, 32, 35]) final_h2.shape = ([2, 32, 35])

    def shared_modaties(self, t_feature, v_feature, a_feature): # ([32, 768]) ([32, 140]) ([32, 296]) 

        # # (2-1) Version1 模态特定(单独学习) # such as ([32, 296]) -> ([2, 32, 296]) -> ([2, 32, 296])
        # A_1, A_2 = self.tf_encoder_private_a(torch.cat([torch.unsqueeze(a_feature,0),torch.unsqueeze(a_feature,0)],dim=0),torch.cat([torch.unsqueeze(a_feature,0),torch.unsqueeze(a_feature,0)],dim=0))
        # V_1, V_2 = self.tf_encoder_private_v(torch.cat([torch.unsqueeze(v_feature,0),torch.unsqueeze(v_feature,0)],dim=0),torch.cat([torch.unsqueeze(v_feature,0),torch.unsqueeze(v_feature,0)],dim=0))
        # T_1, T_2 = self.tf_encoder_private_t(torch.cat([torch.unsqueeze(t_feature,0),torch.unsqueeze(t_feature,0)],dim=0),torch.cat([torch.unsqueeze(t_feature,0),torch.unsqueeze(t_feature,0)],dim=0))
        # self.private_a, self.private_v, self.private_t = (A_1+A_2)/2.0, (V_1+V_2)/2.0, (T_1+T_2)/2.0

        # (2-1) Version2 模态特定(交互学习) 模态特定 # ([32, 296]) -> ([2, 32, 296]) -> ([2, 32, 296])
        A_1, V_2 = self.tf_encoder_private_a(torch.cat([torch.unsqueeze(a_feature,0),torch.unsqueeze(a_feature,0)],dim=0),torch.cat([torch.unsqueeze(v_feature,0),torch.unsqueeze(v_feature,0)],dim=0))
        V_1, T_2 = self.tf_encoder_private_v(torch.cat([torch.unsqueeze(v_feature,0),torch.unsqueeze(v_feature,0)],dim=0),torch.cat([torch.unsqueeze(t_feature,0),torch.unsqueeze(t_feature,0)],dim=0))
        T_1, A_2 = self.tf_encoder_private_t(torch.cat([torch.unsqueeze(t_feature,0),torch.unsqueeze(t_feature,0)],dim=0),torch.cat([torch.unsqueeze(a_feature,0),torch.unsqueeze(a_feature,0)],dim=0))
        self.private_a, self.private_v, self.private_t = (A_1+A_2)/2.0, (V_1+V_2)/2.0, (T_1+T_2)/2.0

        # (2-2) 模态绑定
        # Projecting to same sized space
        t_feature = self.project_t(t_feature) # ([32, 768]) -> ([32, 256]) # 保留独立模态
        v_feature = self.project_v(v_feature) # ([32, 140]) -> ([32, 256])
        a_feature = self.project_a(a_feature) # ([32, 296]) -> ([32, 256])
        try: # 防止技术维度的特征绑定失衡 如([27, 256]) 绑定后变为 ([32, 256]) 最后一个batch不足32个样本
            self.t_feature = torch.cat((t_feature.chunk(2, dim=0)[0], v_feature.chunk(2, dim=0)[0]),dim=0) # 模态绑定
            self.v_feature = torch.cat((v_feature.chunk(2, dim=0)[0], a_feature.chunk(2, dim=0)[0]),dim=0)
            self.a_feature = torch.cat((a_feature.chunk(2, dim=0)[0], t_feature.chunk(2, dim=0)[0]),dim=0)
            # (2-3) 模态不变(共享学习)
            # 采用改进的Transformer进行模态不变学习，同时 输入 text, video, audio 三个模态分别作为 Q K V 计算，输出交互后的模态特征 (每个模态都由 绑定模态 和 独立模态 组成)
            self.share_T, self.share_V, self.share_A = self.tf_encoder_share(torch.stack([self.t_feature,t_feature],dim=0), torch.stack([self.v_feature,v_feature],dim=0), torch.stack([self.a_feature,a_feature],dim=0)) # ([2, 32, 256]) -> ([2, 32, 256])
        except:
            self.t_feature = t_feature + v_feature
            self.v_feature = v_feature + a_feature
            self.a_feature = a_feature + t_feature
            self.share_T, self.share_V, self.share_A = self.tf_encoder_share(torch.stack([self.t_feature,t_feature],dim=0), torch.stack([self.v_feature,v_feature],dim=0), torch.stack([self.a_feature,a_feature],dim=0)) # ([2, 32, 256]) -> ([2, 32, 256])

    def alignment(self, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask): # MOSEI: visual([59, 32, 35]) & acoustic([59, 32, 74]) & lengths([32]) & ([32, 61])

        batch_size = lengths.shape[0]

        # (1-1) 文本特征提取
        bert_output = self.bertmodel(input_ids=bert_sent, # bert_sent ([64, 41])
                                    attention_mask=bert_sent_mask, # bert_sent_mask ([64, 41])
                                    token_type_ids=bert_sent_type) # bert_sent_type ([64, 41])

        bert_output = bert_output[0] # bert_output[0].shape ([64, 41, 768]) len(bert_output)=3  bert_output[1].shape ([64, 768])
        # masked mean
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output) # ([64, 41])->([64, 41, 1]) @ ([64, 41, 768]) -> masked_output ([64, 41, 768])
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) # mask_len (64,1)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len # 'torch.cuda.FloatTensor' bert_output ([64, 768])
        t_feature = bert_output # ([64, 768])

        # (1-2) 视频特征提取
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm) # ([59, 32, 35]) -> ([2, 32, 35]), ([2, 32, 35])
        video_feature = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 140])
 
        # (1-3) 音频特征提取
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm) # ([59, 32, 74]) -> ([2, 32, 74]), ([2, 32, 74])
        audio_feature = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 296])

        # (2) 模态不变 模态绑定 模态特定
        self.shared_modaties(t_feature, video_feature, audio_feature) # 模态对齐->模态共享 ([32, 256])

        # (3) 采用直接融合的方法
        self.private_a = self.fc_a(self.private_a)  # ([2, 32, 296]) -> ([2, 32, 256])
        self.private_v = self.fc_v(self.private_v)
        self.private_t = self.fc_t(self.private_t)
        # ===== 额外引入一层通信层用来提升性能
        self.private_a = self.MLP_Communicator1(self.private_a.permute(1, 0, 2)).permute(1, 0, 2) # ([2, 32, 296]) -> ([32, 2, 256]) -> ([2, 32, 256]) 为了适应之后的loss计算
        self.private_v = self.MLP_Communicator1(self.private_v.permute(1, 0, 2)).permute(1, 0, 2)
        self.private_t = self.MLP_Communicator1(self.private_t.permute(1, 0, 2)).permute(1, 0, 2)            
        self.share_T = self.MLP_Communicator2(self.share_T.permute(1, 0, 2)).permute(1, 0, 2)
        self.share_V = self.MLP_Communicator2(self.share_V.permute(1, 0, 2)).permute(1, 0, 2)
        self.share_A = self.MLP_Communicator2(self.share_A.permute(1, 0, 2)).permute(1, 0, 2)
        
        # # ===== 加入 可学习的 cls token 区分 6 种模态信息 避免模型偏好某一种特定的模态 (可以提升一点acc7,acc2提升不大)
        # try: # 防止技术维度的特征绑定失衡 如([27, 256]) 绑定后变为 ([32, 256]) 最后一个batch不足32个样本
        #     all_modal = torch.cat([self.private_a.permute(1, 0, 2), self.private_v.permute(1, 0, 2), self.private_t.permute(1, 0, 2), self.cls_token, \
        #                         self.share_T.permute(1, 0, 2), self.share_V.permute(1, 0, 2), self.share_A.permute(1, 0, 2), self.cls_token],dim=1) # 捕获输入的分段信息
        #     all_modal = all_modal + self.pos_embed # 捕获输入的顺序信息
        #     # ([32, 2, 256])*3+([32, 1, 256])+([32, 2, 256])*3+([32, 1, 256]) -> ([32, 14, 256]) [B,L,dim] / [B,C,H,W]
        # except:
        #     all_modal = torch.cat([self.private_a.permute(1, 0, 2), self.private_v.permute(1, 0, 2), self.private_t.permute(1, 0, 2), \
        #                         self.share_T.permute(1, 0, 2), self.share_V.permute(1, 0, 2), self.share_A.permute(1, 0, 2)],dim=1)           
            
        all_modal = torch.cat([self.private_a.permute(1, 0, 2), self.private_v.permute(1, 0, 2), self.private_t.permute(1, 0, 2), \
                            self.share_T.permute(1, 0, 2), self.share_V.permute(1, 0, 2), self.share_A.permute(1, 0, 2)],dim=1)        
                      
        second_o7 = self.fusion3(all_modal.permute(0, 2, 1)).view(all_modal.size(0), -1) 
        
        return second_o7

    def forward(self, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        o = self.alignment(video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
    
def l2_normalize(tensor): # Jack Define 使用np.linalg.norm函数计算数组的L2范数，并将其用于归一化操作
    normalized_tensor = F.normalize(tensor, p=2, dim=1)
    return normalized_tensor

def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features



# =============================
# Adapted From PXMixer
# =============================

from einops.layers.torch import Rearrange

class MLP_block(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP_Communicator(nn.Module):
    def __init__(self, token, channel, hidden_size, depth=1):
        super(MLP_Communicator, self).__init__()
        self.depth = depth
        self.token_mixer = nn.Sequential(
            Rearrange('b n d -> b d n'),
            MLP_block(input_size=channel, hidden_size=hidden_size),
            Rearrange('b n d -> b d n')
        )
        self.channel_mixer = nn.Sequential(
            MLP_block(input_size=token, hidden_size=hidden_size)
        )

    def forward(self, x):
        for _ in range(self.depth):
            x = x + self.token_mixer(x)
            x = x + self.channel_mixer(x)
        return x


