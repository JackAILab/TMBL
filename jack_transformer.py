from torch import nn

import torch
import torch.nn.functional as F
import numpy as np

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
    
class FBP(nn.Module):
    def __init__(self, d_emb_1, d_emb_2, fbp_hid, fbp_k, dropout):
        super(FBP, self).__init__()
        self.fusion_1_matrix = nn.Linear(d_emb_1, fbp_hid*fbp_k, bias=False)
        self.fusion_2_matrix = nn.Linear(d_emb_2, fbp_hid*fbp_k, bias=False)
        self.fusion_dropout = nn.Dropout(dropout)
        self.fusion_pooling = nn.AvgPool1d(kernel_size=fbp_k)
        self.fbp_k = fbp_k

    def forward(self, seq1, seq2):
        seq1 = self.fusion_1_matrix(seq1)
        seq2 = self.fusion_2_matrix(seq2)
        fused_feature = torch.mul(seq1, seq2)
        if len(fused_feature.shape) == 2:
            fused_feature = fused_feature.unsqueeze(0)
        fused_feature = self.fusion_dropout(fused_feature)
        fused_feature = self.fusion_pooling(fused_feature).squeeze(0) * self.fbp_k # (bs, 512)
        fused_feature = F.normalize(fused_feature, dim=-1, p=2)
        return fused_feature
    
class MultiHeadAttention(nn.Module): # MultiHeadAttention(n_head=4, d_emb_q=256, d_emb_v=128, d_k=512, d_v=1024)
    # def __init__(self, n_head, d_emb_q, d_emb_v, d_k=None, d_v=None, dropout=0.1):
    def __init__(self, n_head, d_emb_q, d_emb_v, d_k=512, d_v=1024, dropout=0.1): # Jack Add
        super(MultiHeadAttention, self).__init__()
        self.d_emb_q, self.d_emb_v, self.n_head = d_emb_q, d_emb_v, n_head
        self.d_k = d_k if d_k is not None else d_emb_q
        self.d_v = d_v if d_v is not None else d_emb_v

        assert self.d_k % n_head == 0, 'Error from MultiHeadAttention: self.d_k % n_head should be zero.'
        assert self.d_v % n_head == 0, 'Error from MultiHeadAttention: self.d_v % n_head should be zero.'
        
        self.w_q = nn.Linear(d_emb_q, self.d_k, bias=False)
        self.w_k = nn.Linear(d_emb_v, self.d_k, bias=False)
        self.w_v = nn.Linear(d_emb_v, self.d_v, bias=False)
        self.fc = nn.Linear(self.d_v, d_emb_q, bias=False)

        self.fbp = FBP(self.d_k, self.d_k, 32, 2, dropout)
        self.fc_gate = nn.Linear(32, 1)
        self.gate_activate = nn.Tanh()
        # self.gate_activate = nn.Sigmoid()
        # self.tfn = TFN(
        #     input_dims=[self.d_k, self.d_k, self.d_v], 
        #     hidden_dims=[64, 64, 64], out=[64, 64, 64], 
        #     dropouts=[dropout, dropout, dropout, dropout], 
        #     # dropouts=[0, 0, 0, 0], 
        #     post_fusion_dim=32)
            
        # self.output_thredshole = Parameter(torch.FloatTensor([0]), requires_grad=True)

        # self.w_q = nn.Conv1d(d_emb_q, self.d_k, 3, 1, 1, bias=False)
        # self.w_k = nn.Conv1d(d_emb_v, self.d_k, 3, 1, 1, bias=False)
        # self.w_v = nn.Conv1d(d_emb_v, self.d_v, 3, 1, 1, bias=False)
        # self.fc = nn.Conv1d(self.d_v, d_emb_q, 3, 1, 1, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_emb_q, eps=1e-6)

    def forward(self, q, k, v, mask1=None, mask2=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        assert len_k == len_v, 'len_k should be equal with len_v.'

        residual = q
        # print(q.shape)

        # Separate different heads: b x l x n x (d/n)
        # q = self.w_q(q.transpose(1, 2)).transpose(1, 2).view(sz_b, len_q, n_head, d_k // n_head)
        # k = self.w_k(k.transpose(1, 2)).transpose(1, 2).view(sz_b, len_k, n_head, d_k // n_head)
        # v = self.w_v(v.transpose(1, 2)).transpose(1, 2).view(sz_b, len_v, n_head, d_v // n_head)
        q = self.w_q(q).view(sz_b, len_q, n_head, d_k // n_head)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k // n_head)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_v // n_head)

        # q_, k_ = q.view(sz_b, len_q, d_k).mean(1), k.view(sz_b, len_k, d_k).mean(1)
        # q_, k_, v_ = q.view(sz_b, len_q, d_k), k.view(sz_b, len_k, d_k), v.view(sz_b, len_v, d_v)
        q_, k_ = q.view(sz_b, len_q, d_k), k.view(sz_b, len_k, d_k)
        gate_ = self.fbp(q_, k_)
        # gate_ = self.tfn(q_, k_, v_)
        gate_ = self.gate_activate(self.fc_gate(gate_))#.double()
        # gate_ = self.gate_activate(gate_).double()
        # gate_ = torch.where(gate_ > 0.0, gate_, 0.0).double()
        # gate_ = torch.where(gate_ <=0.0, gate_, 1.0).float()
        # gate_ = torch.where(gate_ > 0.0, 1.0, 0.0).float()
        gate_sign = gate_ / torch.abs(gate_)
        # print(gate_sign.detach().cpu().numpy().tolist())
        gate_ = (gate_sign + torch.abs(gate_sign)) / 2.0
        # print(gate_.detach().cpu().numpy().tolist())
        # print(gate_.requires_grad, gate_.grad_fn, end= ' \n' )

        # print(gate_.requires_grad, gate_.grad_fn, gate_tmp.requires_grad, gate_tmp.grad_fn, end= '\n' )
        # print((gate_>0).float().detach().sum().cpu().numpy() / len(gate_), end=' ')

        # Transpose for attention dot product: b x n x l x d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(1).unsqueeze(-1)  # For head axis broadcasting.
            mask2 = mask2.unsqueeze(1).unsqueeze(2)  # For head axis broadcasting.

        # result b x n x lq x (dv/n)
        result, attn = self.attention(q, k, v, mask1=mask1, mask2=mask2)

        # Transpose to move the head dimension back: b x l x n x (dv/n)
        # Combine the last two dimensions to concatenate all the heads together: b x l x (dv)
        result = result.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        # b x l x (d_model)
        # result = self.dropout(self.fc(result.transpose(1, 2)).transpose(1, 2))
        result = self.dropout(self.fc(result))
        if len(gate_.shape) == 2:
            gate_ = gate_.unsqueeze(-1)
        result = result * gate_
        
        # print(gate_.requires_grad, end= '' )
        # result = result.masked_fill(gate_ < 0, 0)
        result += residual

        result = self.layer_norm(result)
        result = result.masked_fill(torch.isnan(result), 0.0)

        return result, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
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

        # === Jack Add
        desired_dim = 64
        current_shape = self.pos_table.shape
        pad_length = desired_dim - current_shape[1]
        # 计算填充的左侧和右侧长度
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        # 使用 pad 函数在第二个维度上进行填充
        padded_table = F.pad(self.pos_table, (0, 0, left_pad, right_pad))
        # 更新 self.pos_table
        self.pos_table = padded_table
        # === Jack Add

        return x + self.pos_table[:, :x.size(1)].clone().detach()

class EncoderLayer(nn.Module):

    def __init__(self, n_head, d_emb_q, d_emb_v, d_inner, d_k=None, d_v=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_v, d_k=d_k, d_v=d_v, dropout=dropout)
        self.slf_attn_sa = MultiHeadAttention(n_head=n_head, d_emb_q=d_emb_q, d_emb_v=d_emb_q, d_k=d_k, d_v=d_k, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_emb_q, d_inner, dropout=dropout)

    def forward(self, q, k, v, slf_attn_mask1=None, slf_attn_mask2=None):
        enc_output, enc_slf_attn = self.slf_attn(q, k, v, mask1=slf_attn_mask1, mask2=slf_attn_mask2)
        enc_output, enc_slf_attn = self.slf_attn_sa(q, q, q, mask1=slf_attn_mask1, mask2=slf_attn_mask2)
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
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.layer_norm1 = nn.LayerNorm(d_emb_1, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_emb_2, eps=1e-6)

        self.layer_stack1 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_1, d_emb_2, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_stack2 = nn.ModuleList([
            EncoderLayer(n_head, d_emb_2, d_emb_1, d_inner, d_k, d_out, dropout=dropout)
            for _ in range(n_layers)])


    def forward(self, seq1, seq2, src_mask1=None, src_mask2=None, return_attns=False):
        enc_slf_attn_list1, enc_slf_attn_list2 = [], []

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
