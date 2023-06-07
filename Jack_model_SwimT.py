import torch
import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
from einops.layers.torch import Rearrange
from swin_transformer import SwinTransformer, load_pretrained, create_logger
import math

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
        # swim Transformer
        ##########################################
        self.swinT_hidden_size = config.hidden_size
        layernorm = nn.LayerNorm
        logger = create_logger(output_dir="/home/jack/Project/MutiModal/SentimentAnalysis/JackNet/output/", name='swin_base_patch4_window7_224')
        self.swimT = SwinTransformer(img_size= 224, # DATA.IMG_SIZE
                                patch_size= 4, # MODEL.SWIN.PATCH_SIZE
                                in_chans= 3, # MODEL.SWIN.IN_CHANS
                                num_classes= 1000, # MODEL.NUM_CLASSES
                                embed_dim= 128, # MODEL.SWIN.EMBED_DIM
                                depths= [ 2, 2, 18, 2 ], # MODEL.SWIN.DEPTHS
                                num_heads= [ 4, 8, 16, 32 ], # MODEL.SWIN.NUM_HEADS
                                window_size= 7, # MODEL.SWIN.WINDOW_SIZE
                                mlp_ratio= 4, # MODEL.SWIN.MLP_RATIO
                                qkv_bias= True, # MODEL.SWIN.QKV_BIAS
                                qk_scale= None, # MODEL.SWIN.QK_SCALE
                                drop_rate= 0.0, # MODEL.DROP_RATE
                                drop_path_rate= 0.1, # MODEL.DROP_PATH_RATE
                                ape= False, # MODEL.SWIN.APE
                                norm_layer=layernorm,
                                patch_norm= True, # MODEL.SWIN.PATCH_NORM
                                use_checkpoint= False, # TRAIN.USE_CHECKPOINT
                                fused_window_process=False # FUSED_WINDOW_PROCESS
        )
        self.swimT_up = nn.UpsamplingNearest2d(scale_factor=14)
        self.swimT_fusion = nn.Sequential()
        self.swimT_fusion.add_module('fusion_layer_1', nn.Linear(in_features=1000,
                                                           out_features=6 * self.config.hidden_size, bias = False))
        self.swimT_fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.swimT_fusion.add_module('fusion_layer_1_activation', self.activation)
        self.swimT_fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=6 * self.config.hidden_size, out_features=output_size, bias = False))
        # load_pretrained(self.swimT, logger)
        
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

        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)

        bert_output = bert_output[0]

        # masked mean
        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = bert_output # ([32, 768])

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm) # ([59, 32, 35]) -> ([2, 32, 35]), ([2, 32, 35])
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 140])

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm) # ([59, 32, 74]) -> ([2, 32, 74]), ([2, 32, 74])
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1) # ([32, 296])

        # Shared-private encoders
        self.shared_modaties(batch_size, self.swinT_hidden_size, utterance_text, utterance_video, utterance_audio) # 模态对齐->模态共享 ([32, 128]) 

        # h1 = torch.stack((self.utt_shared_v, self.utt_shared_t), dim=0) # h1.shape = ([2, 32, 128])
        # h2 = torch.stack((self.utt_shared_a2, self.utt_shared_t2), dim=0) # h2.shape = ([2, 32, 128])

        # h1 = self.batchnorm(h1.permute(1, 0, 2)) # h1.shape = ([32, 2, 128])
        # h2 = self.batchnorm(h2.permute(1, 0, 2))

        # # h1 = self.transformer_encoder1(h1).permute(1, 0, 2)
        # # h2 = self.transformer_encoder2(h2).permute(1, 0, 2)

        # h1 = self.MLP_Communicator1(h1).permute(1,0,2) # h1.shape = ([2, 32, 128])
        # h2 = self.MLP_Communicator2(h2).permute(1,0,2)

        # # h1 = h1.permute(2, 0, 1)
        # # h2 = h2.permute(2, 0, 1)
        # h1 = torch.cat((h1[0], h1[1]), dim=1) # ([32, 256])
        # h2 = torch.cat((h2[0], h2[1]), dim=1)

        # norm1 = torch.norm(h1, dim = 1,p=1) # ([32])
        # norm2 = torch.norm(h2, dim = 1,p=1)

        # self.scale = norm2
        # self.polar_vector = h1

        # h1 = h1 * (torch.div(norm2.unsqueeze(1), norm1.unsqueeze(1))) # h1*投影比例(0~1).shape=([32, 1])

        # o7 = self.fusion(h1) # ([32, 256]) -> ([32, 1])

        # o7_1 = self.fusion(h1) # ([32, 256]) -> ([32, 1])
        # o7_2 = self.swimT_fusion(self.swim_share_modal) # ([32, 1000]) -> ([32, 1]) 
        # o7 = o7_1 + o7_2

        o7 = self.swimT_fusion(self.swim_share_modal) # ([32, 1000]) -> ([32, 1])
        return o7

    def shared_modaties(self, batch_size, SwinT_hidden_size, utterance_t, utterance_v, utterance_a):

        # Projecting to same sized space
        utterance_t = self.project_t(utterance_t) # ([32, 768]) -> ([32, 128])
        utterance_v = self.project_v(utterance_v) # ([32, 140]) -> ([32, 128])
        utterance_a = self.project_a(utterance_a) # ([32, 296]) -> ([32, 128]) 
        
        ##########################################
        # swim Transformer Start
        ##########################################
        multimoda = torch.stack((utterance_t.reshape(batch_size,int(math.sqrt(SwinT_hidden_size)),int(math.sqrt(SwinT_hidden_size))), \
                                 utterance_v.reshape(batch_size,int(math.sqrt(SwinT_hidden_size)),int(math.sqrt(SwinT_hidden_size))), \
                                utterance_a.reshape(batch_size,int(math.sqrt(SwinT_hidden_size)),int(math.sqrt(SwinT_hidden_size)))),dim=1) # ([32, 3, 16, 16])
        multimoda = self.swimT_up(multimoda) # ([32, 3, 16, 16]) ->  ([32, 3, 224, 224])
        self.swim_share_modal = self.swimT(multimoda)
        ##########################################
        # swim Transformer End
        ##########################################

        # self.utt_shared_t = self.shared1(utterance_t) # ([32, 128]) -> ([32, 128]) 
        # self.utt_shared_v = self.shared1(utterance_v) # ([32, 128]) -> ([32, 128]) 

        # self.utt_shared_t2 = self.shared2(utterance_t) # ([32, 128]) -> ([32, 128]) 
        # self.utt_shared_a2 = self.shared2(utterance_a) # ([32, 128]) -> ([32, 128]) 

    def forward(self, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        o = self.alignment(video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o

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























