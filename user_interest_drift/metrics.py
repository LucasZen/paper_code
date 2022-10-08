import torch
import numpy as np

'''
from torch import nn
def cal_loss(scores, targets, loss_function):
    targets = targets.view(-1)
    scores = scores.view(-1, scores.size(-1))  # (B*T) x V
    scores = scores.cpu()
    loss = loss_function(scores, targets)
    return loss 
'''
    
def Metrics(model, loader_test, top_ks):
    '''
    loss_total = 0
    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    '''
    user_number = 0
    HRs = [0] * len(top_ks)
    MRRs = [0] * len(top_ks)
    NDCGs = [0] * len(top_ks)
    for data, neigh, y in loader_test:
        with torch.no_grad():
            data = data.cuda()
            neigh = neigh.cuda()
            scores = model(data, neigh)
            '''
            loss = cal_loss(scores, y, loss_function)
            loss_total += loss.item()
            '''
        y = y.view(-1)
        for pos, top_k in enumerate(top_ks):
            hit = HRs[pos]; mrr = MRRs[pos]; ndcg = NDCGs[pos]
            scores = scores.view(-1, scores.size(-1))  # (B*T) x V
            _, top_k_item_pos = torch.topk(scores, top_k)
            top_k_item_pos = top_k_item_pos.cpu()

            for next_item_pos, top_k_item_pos_per in zip(y, top_k_item_pos):
                if next_item_pos != -1:
                    if next_item_pos in top_k_item_pos_per:
                        hit += 1
                    ndcg += DCG(top_k_item_pos_per, next_item_pos)
                    mrr += MRR(top_k_item_pos_per, next_item_pos)
                    user_number += 1
            HRs[pos] = hit; MRRs[pos] = mrr; NDCGs[pos] = ndcg

    HR = [i / user_number for i in HRs]
    Mrr = [i / user_number for i in MRRs]
    NDCG = [i / user_number for i in NDCGs]
    return HR, NDCG, Mrr # , loss_total

def DCG(top_k_item_pos_per, next_item_pos):
    if next_item_pos in top_k_item_pos_per:
        top_k_item_pos_per = top_k_item_pos_per.detach().numpy().tolist()
        index = top_k_item_pos_per.index(next_item_pos)
        return np.reciprocal(np.log2(index + 2))
    else:
        return 0

def MRR(top_k_item_pos_per, next_item_pos):
    if next_item_pos in top_k_item_pos_per:
        top_k_item_pos_per = top_k_item_pos_per.detach().numpy().tolist()
        index = top_k_item_pos_per.index(next_item_pos)
        return 1 / (index + 1)
    else:
        return 0