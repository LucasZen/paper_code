import torch
from torch import nn
from BERT.Muti_self_attention import MultiHeadedAttention
from BERT.feed_forward import PositionwiseFeedForward
from BERT.res_net import SublayerConnection

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        # attention unit layer
        self.attention_unit = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def attention_unit_layer(self, x, all_user_nei):
        attention = []
        for i in range(all_user_nei.shape[1]):
            nei = all_user_nei[:, i, :]
            nei = nei.unsqueeze(dim=1)
            nei = nei.repeat(1, x.shape[1], 1)
            new_info = nei - x
            rel = x * nei
            merge_info = torch.cat((x, rel, new_info, nei), dim=-1)
            att = self.attention_unit(merge_info)
            attention.append(att)
        x_nei_att = torch.cat(attention, dim=-1)
        x_nei_att = nn.Softmax(dim=-1)(x_nei_att)
        x = x + x_nei_att @ all_user_nei
        return x
    def forward(self, x, mask, neighs=None, neighs_mask=None):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # '''
        x_neigh = torch.Tensor([]).cuda()
        for neigh, neigh_mask in zip(neighs, neighs_mask):
            neigh = self.input_sublayer(neigh, lambda _x: self.attention.forward(_x, _x, _x, mask=neigh_mask))
            last = neigh[:, -1, :].unsqueeze(dim=1)
            x_neigh = torch.cat((x_neigh, last), dim=1)
        # x_neigh_att = nn.Softmax(dim=2)(x @ torch.transpose(x_neigh, dim0=1, dim1=2))
        # x = x + x_neigh_att @ x_neigh
        # x = self.attention_unit_layer(x, x_neigh) # origin_pos to 58 row 
        # '''

        x = self.output_sublayer(x, self.feed_forward)
        x = self.attention_unit_layer(x, x_neigh) # here
        return self.dropout(x)
