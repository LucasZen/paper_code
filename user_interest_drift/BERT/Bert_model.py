import torch
from torch import nn
from BERT.Bert_embedding import BERTEmbedding
from BERT.TransformBlock import TransformerBlock
from BERT.random_seed import fix_random_seed_as

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        # fix_random_seed_as()
        self.max_len = args.bert_max_len
        self.n_layers = args.bert_num_blocks
        self.heads = args.bert_num_heads
        self.hidden = args.bert_hidden_units
        self.dropout = args.bert_dropout
        self.vocab_size = args.num_items + 1

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(args, vocab_size=self.vocab_size, embed_size=self.hidden, max_len=self.max_len, dropout=self.dropout)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden, self.heads, self.hidden * 4, self.dropout) for _ in range(self.n_layers)])
        # self.init_weights()

    def forward(self, x, neighs):
        mask = (x < self.vocab_size - 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        x = self.embedding(x)

        # '''
        mask_neighs = []
        for i in range(neighs.shape[1]):
            neigh = neighs[:, i, :]
            neigh_mask = (neigh < self.vocab_size - 1).unsqueeze(1).repeat(1, neigh.size(1), 1).unsqueeze(1)
            mask_neighs.append(neigh_mask)

        neighs_em = []
        for i in range(neighs.shape[1]):
            neigh = neighs[:, i, :]
            neigh = self.embedding(neigh)
            neighs_em.append(neigh)
        # '''
        # neighs_em = None; mask_neighs = None

        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask, neighs_em, mask_neighs)

        return x

    def init_weights(self):
        pass






















