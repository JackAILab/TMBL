from torch import nn

import torch
import torch.nn.functional as F
import numpy as np
"""
Adapted from https://github.com/kiva12138/MITRL/NavieTransformerExpr # Jack Add some different
"""

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask1=None, mask2=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask1 is not None and mask2 is not None:
            # print('Attention', attn.shape, 'Mask', mask1.shape)
            # print('Attention', attn.shape, 'Mask', mask2.shape)
            # # print(mask)
            attn = attn.masked_fill_(mask1, 1e-9) # Fills elements of att with 1e-9 where mask is True.
            attn = attn.masked_fill_(mask2, 1e-9) # Fills elements of att with 1e-9 where mask is True.
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn
    
class MultiHeadAttention(nn.Module): # MultiHeadAttention(n_head=4, d_emb_q=256, d_emb_v=128, d_k=512, d_v=1024)
    # def __init__(self, n_head, d_emb_q, d_emb_v, d_k=None, d_v=None, dropout=0.1):
    def __init__(self, n_head, d_emb_q, d_emb_v, d_k=512, d_v=1024, dropout=0.1): # Jack Add
        super(MultiHeadAttention, self).__init__()
        self.d_emb_q, self.d_emb_v, self.n_head = d_emb_q, d_emb_v, n_head
        self.d_k = d_k if d_k is not None else d_emb_q
        self.d_v = d_v if d_v is not None else d_emb_v

        assert self.d_k % n_head == 0, 'Error from MultiHeadAttention: self.d_k % n_head should be zero.'
        assert self.d_v % n_head == 0, 'Error from MultiHeadAttention: self.d_v % n_head should be zero.'
        
        self.w_q = nn.Linear(d_emb_q, self.d_k, bias=False) # you can also try nn.Conv1d(d_emb_q, self.d_k, 3, 1, 1, bias=False)     d_emb_q=296    self.d_k=32
        self.w_k = nn.Linear(d_emb_v, self.d_k, bias=False) # d_emb_v=296    self.d_k=32
        self.w_v = nn.Linear(d_emb_v, self.d_v, bias=False) # d_emb_v=296    self.d_k=32
        self.fc = nn.Linear(self.d_v, d_emb_q, bias=False)

        # =========== Jack Add Keep Dim 这样即能keep_dim 也可以进行特征共享
        self.keep_q_k = nn.Conv1d(self.d_k, self.d_k, kernel_size=1, stride=1) # self.d_k=32 self.d_k=32
        # =========== Jack Add Keep Dim

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_emb_q, eps=1e-6)

    def forward(self, q, k, v, mask1=None, mask2=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        assert len_k == len_v, 'len_k should be equal with len_v.'

        residual = q

        # Separate different heads: b x l x n x (d/n)
        try:
            q = self.keep_q_k(self.w_q(q)).view(sz_b, len_q, n_head, d_k // n_head) # Jack Add Q K 进行keepdim 并共享注意力参数
            k = self.keep_q_k(self.w_k(k)).view(sz_b, len_k, n_head, d_k // n_head)
            # q = self.w_q(q).view(sz_b, len_q, n_head, d_k // n_head) # Original
            # k = self.w_k(k).view(sz_b, len_k, n_head, d_k // n_head)                        
        except:
            q = self.w_q(q).view(sz_b, len_q, n_head, d_k // n_head) # Original 避免最后一个batch只有27而报错
            k = self.w_k(k).view(sz_b, len_k, n_head, d_k // n_head)                
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v // n_head)

        # Transpose for attention dot product: b x n x l x d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # For head axis broadcasting.
            mask2 = mask2.unsqueeze(1).unsqueeze(2)  # For head axis broadcasting.

        # result b x n x lq x (dv/n)
        result, attn = self.attention(q, k, v, mask1=mask1, mask2=mask2) # result.shape=([2, 8, 48, 4])  attn.shape=torch.Size([2, 8, 48, 48])

        # Transpose to move the head dimension back: b x l x n x (dv/n)
        # Combine the last two dimensions to concatenate all the heads together: b x l x (dv)
        result = result.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # result.shape=([2, 48, 32])

        # b x l x (d_model)
        result = self.dropout(self.fc(result)) # result.shape=([2, 48, 140])

        result += residual # result = result.masked_fill(gate_ < 0, 0)

        result = self.layer_norm(result)
        result = result.masked_fill(torch.isnan(result), 0.0)

        return result, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise d_in=296 d_hid=512
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        # Jack Add Chunck Dim
        self.up_dim = nn.Conv1d(d_hid, d_hid*2, kernel_size=1, stride=1) # 升高一个维度，为chunk做准备
        self.keep_dim = nn.Conv1d(d_hid*2, d_hid*2, kernel_size=1, stride=1) # 维度保持不变
        # Jack Add Chunck Dim

    def forward(self, x): # x.shape ([2, 32, 296])
        residual = x

        # x = self.w_2(F.relu(self.w_1(x))) # Original
        # ============= Jack Add 
        x = self.up_dim(self.w_1(x).permute(0, 2, 1)) # 上升2倍维度 ([2, 32, 512]) -> ([2, 512, 32]) -> ([2, 1024, 32])
        x1, x2 = self.keep_dim(x).chunk(2, dim=1) # 维度保持不变，并将2个维度分离为1个  ([2, 512, 32])
        x = F.relu(x1) * x2 # 两个分离的维度相乘  ([2, 512, 32])
        x = self.w_2(x.permute(0, 2, 1)) # 维度变回原维度 (0601: 考虑数据量大的时候，可以考虑映射到低纬度再操作，会更合理一些，比如 d_in*2  ([2, 512, 32]) -> ([2, 32, 512]) -> ([2, 32, 296])
        # ============= Jack Add 

        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        x = x.masked_fill(torch.isnan(x), 0.0)

        return x

class PositionEncoding(nn.Module):
    def __init__(self, d_hid, n_position=100):
        super(PositionEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(int(pos_i)) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # [1,N,d]

    def forward(self, x):
        # x [B,N,d]
        # print(x.shape ,self.pos_table[:, :x.size(1)].shape)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):

    def __init__(self, n_head, d_emb_q, d_emb_v, d_inner, d_k=None, d_v=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn_sa = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_q, d_k=d_k, d_v=d_k, dropout=dropout)
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_v, d_k=d_k, d_v=d_v, dropout=dropout)        
        self.pos_ffn = PositionwiseFeedForward(d_emb_q, d_inner, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask1=None, slf_attn_mask2=None): 

        # 0527 更改一下策略，使用一种混合 attention 的策略，即针对 q k v三种输入(模态不变)，采用三种，针对q q q一种输入(模态特定)，采用一种
        
        if torch.allclose(k,v): # 按照forward, 只有传入的 k v 会是相等的
            enc_output, enc_slf_attn = self.slf_attn_sa(q, q, q, mask1=slf_attn_mask1, mask2=slf_attn_mask2) # Single Attention
        else:
            enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask1=slf_attn_mask1, mask2=slf_attn_mask2)

        enc_output = self.pos_ffn(enc_output)  
        return enc_output, enc_slf_attn

def mean_temporal(data, aug_dim):
    mean_features = torch.mean(data, dim=aug_dim)
    return mean_features

# MyTransformer(d_a=4*acoustic.size(3), d_v=4*visual.size(3), layers=5, d_inner=512, n_head=8, d_k=32, d_out=64, dropout=0.5, n_position=30, add_sa=True)
class MyTransformer(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, n_layers, d_inner, n_head, d_k=None, d_out=None, dropout=0.1, n_position=2048, add_sa=False):
        super(MyTransformer, self).__init__()

        self.position_enc1 = PositionEncoding(d_emb_1, n_position=n_position)
        self.position_enc2 = PositionEncoding(d_emb_2, n_position=n_position)
        self.position_enc3 = PositionEncoding(d_emb_2, n_position=n_position)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_emb_1, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_emb_2, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_emb_2, eps=1e-6)

        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_stack2 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
        
        self.layer_stack3_1 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_stack3_2 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_stack3_3 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
                
    def forward(self, seq1, seq2, seq3=None, src_mask1=None, src_mask2=None, return_attns=False): # seq3 为 JackAdd 目的是将 三种模态共同输入，分别作为Q K V
        enc_slf_attn_list1, enc_slf_attn_list2, enc_slf_attn_list3 = [], [], []

        # ======================== Jack Add Mask Start ======================================
        # src_mask1 = (torch.sum(torch.abs(seq1), dim=-1) < torch.mean(torch.sum(torch.abs(seq1), dim=-1)))  # 由于特征都是使用bert等模型提取的，所有不存在0值，这可能也是为什么MITRL不行的原因，mask掉这些数据的0值是不强的
        # src_mask2 = (torch.sum(torch.abs(seq2), dim=-1) < torch.mean(torch.sum(torch.abs(seq2), dim=-1))) # torch.mean torch.median() 效果并不理想  A2 0.8352/0.8462 A7 0.487
        # src_mask3 = (torch.sum(torch.abs(seq3), dim=-1) == 0)       
        # ======================== Jack Add Mask End ======================================

        # ======================== Jack Add Start =====================================
        if seq3 is not None:

            enc_output1 = self.layer_norm1(self.dropout1(self.position_enc1(seq1)))
            enc_output2 = self.layer_norm2(self.dropout2(self.position_enc2(seq2)))            
            enc_output3 = self.layer_norm3(self.dropout3(self.position_enc3(seq3))) # 减小参数

            enc_output1 = enc_output1.masked_fill(torch.isnan(enc_output1), 0.0) # 这一块很存疑，怀疑是它把 (test_truth[non_zeros] > 0) 这种 pos/neg 的区分性能损害了
            enc_output2 = enc_output2.masked_fill(torch.isnan(enc_output2), 0.0)
            enc_output3 = enc_output3.masked_fill(torch.isnan(enc_output3), 0.0)

            for enc_layer3_1, enc_layer3_2, enc_layer3_3 in zip(self.layer_stack3_1,self.layer_stack3_2,self.layer_stack3_3):
                temp_enc1, temp_enc2, temp_enc3 = enc_output1, enc_output2, enc_output3
                
                enc_output1, enc_slf_attn1  = enc_layer3_1(temp_enc1, temp_enc2, temp_enc3, slf_attn_mask1=src_mask1, slf_attn_mask2=src_mask2)
                enc_output2, enc_slf_attn2  = enc_layer3_2(temp_enc1, temp_enc2, temp_enc3, slf_attn_mask1=src_mask1, slf_attn_mask2=src_mask2)
                enc_output3, enc_slf_attn3  = enc_layer3_3(temp_enc1, temp_enc2, temp_enc3, slf_attn_mask1=src_mask1, slf_attn_mask2=src_mask2)

                enc_slf_attn_list1 += [enc_slf_attn1] if return_attns else []
                enc_slf_attn_list2 += [enc_slf_attn2] if return_attns else []
                enc_slf_attn_list3 += [enc_slf_attn3] if return_attns else []
                
            if return_attns:
                return enc_output1, enc_output2, enc_output3, enc_slf_attn_list1, enc_slf_attn_list2, enc_slf_attn_list3
            return enc_output1, enc_output2, enc_output3
        
        # ======================== Jack Add End=====================================

        else:

            enc_output1 = self.layer_norm1(self.dropout1(self.position_enc1(seq1)))
            enc_output2 = self.layer_norm2(self.dropout2(self.position_enc2(seq2)))

            enc_output1 = enc_output1.masked_fill(torch.isnan(enc_output1), 0.0)
            enc_output2 = enc_output2.masked_fill(torch.isnan(enc_output2), 0.0)

            for enc_layer1, enc_layer2 in zip(self.layer_stack1, self.layer_stack2):
                temp_enc1, temp_enc2 = enc_output1, enc_output2
                enc_output1, enc_slf_attn1 = enc_layer1(temp_enc1, temp_enc2, temp_enc2, slf_attn_mask1=src_mask1, slf_attn_mask2=src_mask2)
                enc_output2, enc_slf_attn2 = enc_layer2(temp_enc2, temp_enc1, temp_enc1, slf_attn_mask1=src_mask2, slf_attn_mask2=src_mask1)

                enc_slf_attn_list1 += [enc_slf_attn1] if return_attns else []
                enc_slf_attn_list2 += [enc_slf_attn2] if return_attns else []

            if return_attns:
                return enc_output1, enc_output2, enc_slf_attn_list1, enc_slf_attn_list2
            return enc_output1, enc_output2        


def make_mask(feature):
    return torch.sum(torch.abs(feature), dim=-1) == 0


if __name__ == '__main__':
    encoder = MyTransformer(d_emb_1=128, d_emb_2=256, n_layers=1, d_inner=512, n_head=2)
    a = torch.randn(4, 4, 128)
    b = torch.randn(4, 6, 256)
    a_mask, b_mask = make_mask(a), make_mask(b)
    a_mask = torch.Tensor([
        [False, False, False, False],
        [False, False, False, True],
        [False, False, False, True],
        [False, True, True, True],
    ]).long().bool()
    b_mask = torch.Tensor([
        [False, False, False, False, False, False,],
        [False, False, False, False, False, True],
        [False, False, False, False, False, True],
        [False, False, False, True, True, True],
    ]).long().bool()
    y= encoder(a, b, src_mask1=None, src_mask2=None, return_attns=False)
    print([y_.shape for y_ in y])
