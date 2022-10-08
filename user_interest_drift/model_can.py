import os
import torch
from torch import nn
from BERT.Bert_model import BERT

class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel, self).__init__()
        self.bert = BERT(args)
        self.bert_theme_num = args.bert_theme_num
        self.prediction = nn.Linear(self.bert.hidden, args.num_items)
    def forward(self, x):
        x = self.bert(x)
        scores = self.prediction(x)
        return scores

