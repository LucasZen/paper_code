import os
import torch
from torch import nn
from time import ctime
from model import BERTModel
from torch import nn, optim
from metrics import Metrics
from tqdm import tqdm
from time import ctime

# GMF_model = GMF.GMF(user_num, item_num, args.embedding_dim_GMF, args.dropout)
# GMF_model.load_state_dict(torch.load(GMF_model_path))

def train(data_info, args):
    loader_train = data_info['bert_loader_train']
    loader_test = data_info['bert_loader_test']
    lr = data_info['lr']
    epochs = data_info['epochs']
    top_k = data_info['top_k']
    
    gpus= [0, 1, 2]                                  # multi gpu
    torch.cuda.set_device('cuda:{}'.format(gpus[0])) # multi gpu
    model = BERTModel(args).cuda()

    model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])  # multi gpu

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), weight_decay=1e-5, lr=lr) # weight_decay:1e-5, lr:1e-4
                                                                                             # 0.01

    '''
    Bert_model_path = os.path.join(args.bert_save_path, 'bert_con_unmask_Amazon_books_closed_my_model_dui_bi_001.pth')
    checkpoint = torch.load(Bert_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    '''
    ####

    print("training....", ctime())
    best_ndcg=0.; best_hr = 0.; best_mrr = 0.
    # epochs = 1
    # loss_list = []; epoch_list = []
    for epoch in range(epochs):
        # '''
        model.train()
        loss_total = 0
        for data, neigh, target in tqdm(loader_train):
            data = data.cuda()
            neigh = neigh.cuda()
            scores = model(data, neigh)
            targets = target.view(-1)
            scores = scores.view(-1, scores.size(-1))  # (B*T) x V
            scores = scores.cpu()
            loss = loss_function(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        print(epoch, loss_total, ctime())
        # '''

        model.eval()
        i = 0
        HR, NDCG, MRR = Metrics(model, loader_test, top_k)
        '''
        if epoch < 200:
           loss_list.append(loss_value)
           epoch_list.append(epoch)
        else:
           print('='*10)
           print(loss_list)
           print(epoch_list)
           break 
        '''
        print("HR@{}:{}".format(top_k[i], HR[i]), "NDCG@{}:{}".format(top_k[i], NDCG[i]), "MRR@{}:{}".format(top_k[i], MRR[i]))

        if HR[i] > best_hr:
           best_hr = HR[i]
           # '''
           torch.save({'model_state_dict':model.state_dict(),
                       'optimizer_state_dict':optimizer.state_dict()},
                        os.path.join(args.bert_save_path, 'bert_con_unmask_Amazon_books_closed_my_model_dui_bi_0001.pth'))
           # '''
        best_mrr = MRR[i] if MRR[i] > best_mrr else best_mrr
        best_ndcg = NDCG[i] if NDCG[i] > best_ndcg else best_ndcg

        print("best_HR@{}:{}".format(top_k[i], best_hr), "best_NDCG@{}:{}".format(top_k[i], best_ndcg), "bestMRR@{}:{}".format(top_k[i], best_mrr))


