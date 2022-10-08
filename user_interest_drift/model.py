from torch import nn
from BERT.Bert_model import BERT

class BERTModel(nn.Module):
    def __init__(self, args):
        super(BERTModel, self).__init__()
        self.bert = BERT(args)
        self.bert_theme_num = args.bert_theme_num
        # self.prediction = nn.Linear(self.bert.hidden, args.num_items)
        self.prediction = nn.Linear(self.bert.hidden, 35148)  # merge 11315|| yoochoose:14106|Amazon_books: 35148    |merge 9831|| yoochoose:13400|Amazon_books: 17499

    def forward(self, x, neigh):
        x = self.bert(x, neigh)
        scores = self.prediction(x)
        return scores
