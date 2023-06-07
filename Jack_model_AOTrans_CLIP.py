import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from einops.layers.torch import Rearrange
# from jack_transformer import MyTransformer
from jack_transformer_share import MyTransformer
import torch.nn.functional as F
import numpy as np

# let's define a simple model that can deal with multimodal variable length sequence
class PS_Mixer(nn.Module):
    def __init__(self, config):
        super(PS_Mixer, self).__init__()
        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size
        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        # defining modules - two layer bidirectional LSTM with layer norm in between
        
        # Initializing a BERT bert-base-uncased style configuration
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        # define linear for UR_FUNNY text [32, 300] -> [32, 768]
        self.ur_bert = nn.Linear(in_features=300, out_features=768)

        ##########################################
        # mapping modalities to same sized space
        ##########################################

        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
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

        ##########################################
        # shared encoder
        ##########################################

        self.shared1 = nn.Sequential()
        self.shared1.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared1.add_module('shared_activation_1', nn.Sigmoid())

        self.shared2 = nn.Sequential()
        self.shared2.add_module('shared_2', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared2.add_module('shared_activation_2', nn.Sigmoid())

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 2,
                                                           out_features=6 * self.config.hidden_size, bias = False))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=6 * self.config.hidden_size, out_features=output_size, bias = False))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        self.MLP_Communicator1 = MLP_Communicator(self.config.hidden_size, 2, hidden_size=64, depth=1)
        self.MLP_Communicator2 = MLP_Communicator(self.config.hidden_size, 2, hidden_size=64, depth=1)

        self.batchnorm = nn.BatchNorm1d(2, affine=False)

        ##########################################
        # Transformer
        ##########################################
        self.tf_encoder_av = MyTransformer(d_emb_1=4*input_sizes[2], d_emb_2=4*input_sizes[1], n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True) # None -- 512/1024 Jack Change
        self.tf_encoder_at = MyTransformer(d_emb_1=4*input_sizes[2], d_emb_2=768, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        self.tf_encoder_vt = MyTransformer(d_emb_1=4*input_sizes[1], d_emb_2=768, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # 对齐之后，三模态采用一个共享 Transformer, 以文本模态为中心进行设计 
        self.tf_encoder_share = MyTransformer(d_emb_1=self.config.hidden_size, d_emb_2=self.config.hidden_size, n_layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=config.batch_size, add_sa=True)
        # FC
        self.fc_a = nn.Linear(4*input_sizes[2], self.config.hidden_size)
        self.fc_v = nn.Linear(4*input_sizes[1], self.config.hidden_size)
        self.fc_t = nn.Linear(768, self.config.hidden_size)
        # self.trans_fusion = nn.Sequential(nn.Linear(self.config.hidden_size, self.output_size)) # 采用贡献的fusion，减小参数量的同时充分融合2个网络信息
        ##########################################
        # Transformer
        ##########################################
        self.dim = 32
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.ReLU()
            )
        
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, 3, 1, 1),
            nn.ReLU()
            )   
        
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

    def alignment(self, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask): # MOSEI: visual([59, 32, 35]) & acoustic([59, 32, 74]) & lengths([32]) & ([32, 61])

        batch_size = lengths.shape[0]

        if (bert_sent_type == 1).all() and (bert_sent_mask == 1).all(): # 处理UR_FUNNY等直接使用了Glove提取特征的数据集
            bert_output = self.ur_bert(bert_sent) # [32, 300] -> [32, 768]
            
        else:  # 处理MOSI MOSEI等需要bert提取文本特征的数据集
            bert_output = self.bertmodel(input_ids=bert_sent, # bert_sent ([64, 41])
                                        attention_mask=bert_sent_mask, # bert_sent_mask ([64, 41])
                                        token_type_ids=bert_sent_type) # bert_sent_type ([64, 41])

            bert_output = bert_output[0] # bert_output[0].shape ([64, 41, 768]) len(bert_output)=3  bert_output[1].shape ([64, 768])

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output) # ([64, 41])->([64, 41, 1]) @ ([64, 41, 768]) -> masked_output ([64, 41, 768])
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True) # mask_len (64,1)
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len # 'torch.cuda.FloatTensor' bert_output ([64, 768])


        utterance_text = bert_output # ([64, 768])
        ''' UR_FUNNY 提取到的为 torch.Size([47, 32, 300]) 需要直接变为 ([32, 768]) 将 47 这个维度消除掉 
        [[-0.0042, -0.2269,  0.2518,  ..., -0.3572,  0.0630, -0.1368],
        [ 0.0691,  0.0389,  0.4117,  ..., -0.2666,  0.3193, -0.0698],
        [-0.2460, -0.2926,  0.0414,  ..., -0.2327,  0.5835,  0.1047],
        ..., 
        [ 0.2548, -0.0438, -0.2142,  ..., -0.0204,  0.3280,  0.2674]],
       device='cuda:0', grad_fn=<DivBackward0>)               
        '''
        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm) # ([59, 32, 35]) -> ([2, 32, 35]), ([2, 32, 35])
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 140])

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm) # ([59, 32, 74]) -> ([2, 32, 74]), ([2, 32, 74])
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 296])

        # Shared-private encoders
        self.shared_modaties(utterance_text, utterance_video, utterance_audio) # 模态对齐->模态共享 ([32, 256]) 

        # =================== PXMixer 进行的模态通讯，并且使用极性损失和强度损失训练了这个模型
        # h1 = torch.stack((self.utt_shared_v, self.utt_shared_t), dim=0) # h1.shape = ([2, 32, 256])
        # h2 = torch.stack((self.utt_shared_a2, self.utt_shared_t2), dim=0) # h2.shape = ([2, 32, 256])

        # h1 = self.batchnorm(h1.permute(1, 0, 2)) # h1.shape = ([32, 2, 256])
        # h2 = self.batchnorm(h2.permute(1, 0, 2))

        # # h1 = self.transformer_encoder1(h1).permute(1, 0, 2)
        # # h2 = self.transformer_encoder2(h2).permute(1, 0, 2)

        # h1 = self.MLP_Communicator1(h1).permute(1,0,2) 
        # h2 = self.MLP_Communicator2(h2).permute(1,0,2)

        # # h1 = h1.permute(2, 0, 1) # h1.shape = ([2, 32, 512]) 
        # # h2 = h2.permute(2, 0, 1)
        # h1 = torch.cat((h1[0], h1[1]), dim=1) # ([32, 512])
        # h2 = torch.cat((h2[0], h2[1]), dim=1)

        # norm1 = torch.norm(h1, dim = 1,p=1) # ([32])
        # norm2 = torch.norm(h2, dim = 1,p=1)

        # self.scale = norm2
        # self.polar_vector = h1

        # h1 = h1 * (torch.div(norm2.unsqueeze(1), norm1.unsqueeze(1))) # h1.shape=([32, 512])*投影比例(0~1).shape=([32, 1])

        # o7 = self.fusion(h1) # ([32, 512]) -> ([32, 1])

        # ==== Jack Add 使用 AOTrans 获得的对齐多模态 对 share_modality 进行压缩 Version 1 采用 两两 share 的策略
        # h1 = self.share_T
        # h2 = self.share_V
        # h1 = torch.cat((h1[0], h1[1]), dim=1) # ([32, 512])
        # h2 = torch.cat((h2[0], h2[1]), dim=1)
        # norm1 = torch.norm(h1, dim = 1,p=1) # ([32])
        # norm2 = torch.norm(h2, dim = 1,p=1)
        # self.scale = norm2
        # self.polar_vector = h1
        # h1 = h1 * (torch.div(norm2.unsqueeze(1), norm1.unsqueeze(1))) # h1.shape=([32, 256])*投影比例(0~1).shape=([32, 1])
        # ==== Jack Add 使用 AOTrans 获得的对齐多模态 对 share_modality 进行压缩

        # ==== Jack Add 使用 AOTrans 获得的对齐多模态 对 share_modality 进行压缩 Version 2 采用 三个共同 share 的策略  直接这样跑下来效果不理想 -->> 考虑残差式设计
        if self.dim is not utterance_text.shape[0]:
            h1 = self.share_V + self.share_T # h1.shape = ([2, 32, 256])
            h2 = self.share_A + self.share_T # h2.shape = ([2, 32, 256])
        else:
            res = self.share_V.unsqueeze(3)
            h1 = F.relu(self.res_conv1(self.share_T.unsqueeze(3)) + res)# h1.shape = ([2, 32, 256, 1])
            res = h1
            h1 = F.relu(self.res_conv1(self.share_T.unsqueeze(3)) + res)# h1.shape = ([2, 32, 256, 1])
            h1 = h1.squeeze() + self.share_T # ([2, 32, 256])

            res = self.share_A.unsqueeze(3)
            h2 = F.relu(self.res_conv2(self.share_T.unsqueeze(3)) + res)# h1.shape = ([2, 32, 256])
            res = h2
            h2 = F.relu(self.res_conv2(self.share_T.unsqueeze(3)) + res)# h1.shape = ([2, 32, 256])
            h2 = h2.squeeze() + self.share_T

        h1 = self.batchnorm(h1.permute(1, 0, 2)) # h1.shape = ([32, 2, 256])
        h2 = self.batchnorm(h2.permute(1, 0, 2))
        h1 = self.MLP_Communicator1(h1).permute(1,0,2) 
        h2 = self.MLP_Communicator2(h2).permute(1,0,2)

        h1 = torch.cat((h1[0], h1[1]), dim=1) # ([32, 512])
        h2 = torch.cat((h2[0], h2[1]), dim=1)

        norm1 = torch.norm(h1, dim = 1,p=1) # ([32])
        norm2 = torch.norm(h2, dim = 1,p=1)

        self.scale = norm2
        self.polar_vector = h1        

        h1 = h1 * (torch.div(norm2.unsqueeze(1), norm1.unsqueeze(1))) # h1.shape=([32, 256])*投影比例(0~1).shape=([32, 1])
        # ==== Jack Add 使用 AOTrans 获得的对齐多模态 对 share_modality 进行压缩

        o7_1 = self.fusion(h1) # ([32, 512]) -> ([32, 1])
        o7_2 = self.fusion(self.multimodal)
        o7 = o7_1 + o7_2

        return o7

    def shared_modaties(self, utterance_t, utterance_v, utterance_a):

        # (1) 模态特定 # ([32, 296]) -> ([2, 32, 296]) -> ([2, 32, 296])
        A_1, V_1 = self.tf_encoder_av(torch.cat([torch.unsqueeze(utterance_a,0),torch.unsqueeze(utterance_a,0)],dim=0),torch.cat([torch.unsqueeze(utterance_v,0),torch.unsqueeze(utterance_v,0)],dim=0))
        A_2, T_1 = self.tf_encoder_at(torch.cat([torch.unsqueeze(utterance_a,0),torch.unsqueeze(utterance_a,0)],dim=0),torch.cat([torch.unsqueeze(utterance_t,0),torch.unsqueeze(utterance_t,0)],dim=0))
        V_2, T_2 = self.tf_encoder_vt(torch.cat([torch.unsqueeze(utterance_v,0),torch.unsqueeze(utterance_v,0)],dim=0),torch.cat([torch.unsqueeze(utterance_t,0),torch.unsqueeze(utterance_t,0)],dim=0))
        # A, V, T = A_1+A_2, V_1+V_2, T_1+T_2
        A, V, T = (A_1+A_2)/2.0, (V_1+V_2)/2.0, (T_1+T_2)/2.0

        # Dimensional
        A_F_ = self.fc_a(A).permute(2, 0, 1) # ([2, 16, 296]) -> ([2, 16, 256]) -> ([256, 2, 16])
        V_F_ = self.fc_v(V).permute(2, 0, 1)
        T_F_ = self.fc_t(T).permute(2, 0, 1)
        
        # Out
        # A_F, V_F, T_F = mean_temporal(A_F_, 1), mean_temporal(V_F_, 1), mean_temporal(T_F_, 1)
        # multimodal_feature = torch.stack([A_F, V_F, T_F], dim=-1).sum(dim=-1)
        multimodal_feature = torch.stack([A_F_, V_F_, T_F_], dim=-1).sum(dim=-1) # ([256, 2, 32]) + ([256, 2, 32]) + ([256, 2, 32])
        multimodal_feature = multimodal_feature.view(multimodal_feature.size(0),multimodal_feature.size(1)*multimodal_feature.size(2)).permute(1,0) # torch.Size([256, 2, 32) -> ([256, 64]) -> ([64, 256])
        chunk = torch.chunk(multimodal_feature, chunks=2, dim=0) #  ([64, 256]) -> ([32, 256])
        self.multimodal  = torch.cat([chunk[0], chunk[1]], dim=1) # ([32, 512])

        # (2) 模态绑定-特征图检索
        # Projecting to same sized space
        utterance_t = self.project_t(utterance_t) # ([32, 768]) -> ([32, 256])
        utterance_v = self.project_v(utterance_v) # ([32, 140]) -> ([32, 256])
        utterance_a = self.project_a(utterance_a) # ([32, 296]) -> ([32, 256])         
        # I_e = l2_normalize(np.dot(utterance_t, utterance_v),axis=1) # CLIP 的实现需要引入 nn.Parameter 
        T_V_Weight = l2_normalize(torch.matmul(utterance_v.transpose(0, 1),utterance_t)) # 这里先给个简单的实现，使用 T V 两个模态计算一次权重特征检索图 Weight (256,256) 
        self.utterance_t = utterance_t @ T_V_Weight # (32,256) @ (256,256) 这里关联后的特征图(cos(self.utterance_t,self.utterance_v))会进行一次相似性计算，让文本和视频之间具有更强的关联性
        self.utterance_v = utterance_v @ T_V_Weight
        utterance_t = self.utterance_t # 将检索后的特征赋值回去
        utterance_v = self.utterance_v

        # (3) 模态不变
        # # 采用Transformer进行模态不变的共享学习, 以text特征为基础，输入特征为 v_t 和 a_t
        # self.share_V, self.share_A = self.tf_encoder_share(torch.stack([utterance_v,utterance_t],dim=0), torch.stack([utterance_a,utterance_t],dim=0)) # ([2, 32, 256]) -> ([2, 32, 256])
        # # 采用Transformer进行模态不变的共享学习, 以video特征为基础，输入特征为 t_v 和 a_v
        # self.share_T2, self.share_A2 = self.tf_encoder_share(torch.stack([utterance_t,utterance_v],dim=0), torch.stack([utterance_a,utterance_v],dim=0)) # ([2, 32, 256]) -> ([2, 32, 256])

        # 采用改进的Transformer进行模态不变学习，同时 输入 text, video, audio 三个模态分别作为Q K V 计算，输出交互后的模态特征 (A是由三模态交互得到的，T和V是由两个模态交互得到的)
        # self.share_T, self.share_V, self.share_A = self.tf_encoder_share(torch.stack([utterance_t,utterance_v],dim=0), torch.stack([utterance_a,utterance_v],dim=0), torch.stack([utterance_t,utterance_a],dim=0)) # ([2, 32, 128]) -> ([2, 32, 128])
        self.share_T, self.share_V, self.share_A = self.tf_encoder_share(torch.stack([utterance_t,utterance_t],dim=0), torch.stack([utterance_v,utterance_v],dim=0), torch.stack([utterance_a,utterance_a],dim=0)) # ([2, 32, 128]) -> ([2, 32, 128])

        # self.utt_shared_t = self.shared1(utterance_t) # ([32, 128]) -> ([32, 256]) 
        # self.utt_shared_v = self.shared1(utterance_v) # ([32, 128]) -> ([32, 256]) 

        # self.utt_shared_t2 = self.shared2(utterance_t) # ([32, 128]) -> ([32, 256]) 
        # self.utt_shared_a2 = self.shared2(utterance_a) # ([32, 128]) -> ([32, 256]) 

    def forward(self, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        o = self.alignment(video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o
    
def l2_normalize(tensor): # Jack Define 使用np.linalg.norm函数计算数组的L2范数，并将其用于归一化操作
    normalized_tensor = F.normalize(tensor, p=2, dim=1)
    return normalized_tensor

def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features

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























