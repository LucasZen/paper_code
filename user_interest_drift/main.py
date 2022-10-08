import os
import argparse
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from train import *
from utils import get_train_test_data
import torch.utils.data as Data
from BERT.Dataset import BertTrainDataset, BertTestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Amazon_Books', choices=['LFM-1b', 'yoochoose_visit', 'Amazon_Books'],help='dataset for training')
parser.add_argument("--token", type=str, default='random', choices=['random', 'add_target'], help="random or add_target")
parser.add_argument("--p", type=float, default=0.9, choices=[0.8, 0.9], help="split ratio for train and test")
parser.add_argument("--lr", type=float, default=0.0001, choices=[0.001, 0.0001], help="learning rate")
parser.add_argument("--batch_size", type=int, default=512, help="batch size for training")
parser.add_argument("--epochs", type=int, default=1000, help="training epoches")
parser.add_argument("--top_k", type=list, default=[5], help="compute metrics@top_k")
parser.add_argument("--seed", type=int, default=2022, help="seed")
parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
# bert paramters
parser.add_argument('--bert_max_len', type=int, default=100, help='Length of sequence for bert')
parser.add_argument('--bert_num_items', type=int, default=27278, help='Number of total items')
parser.add_argument('--bert_hidden_units', type=int, default=128, choices=[128, 768], help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=6, choices=[6, 12], help='Number of transformer layers')
parser.add_argument('--bert_num_heads', type=int, default=4, choices=[4, 12], help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.1, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_mask_prob', type=float, default=0.15, help='Probability for masking items in the training sequence')
parser.add_argument('--bert_theme_num', type=int, default=3, choices=[3], help='number of theme')
parser.add_argument('--bert_save_path', type=str, default='BERT/save_model', help='save_path for Bert_model_paramters')
args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


torch.manual_seed(args.seed)  # cpu
torch.cuda.manual_seed(args.seed)  # gpu
np.random.seed(args.seed)  # numpy
random.seed(args.seed)  # random and transforms
torch.backends.cudnn.deterministic = True  # cudnn

if __name__ == '__main__':
    train_data, train_neigh, test_data, test_neigh, item_num, max_len_neigh, max_len_session \
                                       = get_train_test_data(args.dataset, args.token, args.p)

    # Bert paramter
    args.mask_prob = 0.15
    args.mask_token = item_num
    args.num_items = item_num
    args.bert_max_len = max_len_session
    args.bert_max_len_session = max_len_session
    args.bert_max_len_neigh = max_len_neigh
    rng = random.Random(args.seed)

    bert_dataset_train = BertTrainDataset(train_data, train_neigh, args.bert_max_len_session, args.bert_max_len_neigh, args.mask_prob,
                                          args.mask_token, item_num, rng)
    bert_loader_train = Data.DataLoader(bert_dataset_train, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    bert_dataset_test = BertTestDataset(test_data, test_neigh, args.bert_max_len_session, args.bert_max_len_neigh, args.mask_prob,
                                          args.mask_token, item_num, rng)
    bert_loader_test = Data.DataLoader(bert_dataset_test, args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    data_info = {}
    data_info['bert_loader_train'] = bert_loader_train
    data_info['bert_loader_test'] = bert_loader_test

    data_info['item_num'] = item_num
    data_info['lr'] = args.lr
    data_info['batch_size'] = args.batch_size
    data_info['epochs'] = args.batch_size
    data_info['top_k'] = args.top_k
    train(data_info, args)
