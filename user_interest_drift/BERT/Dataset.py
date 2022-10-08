import torch
from torch.utils.data import Dataset

class BertTrainDataset(Dataset):
    def __init__(self, data, neigh_s, max_len_session, max_len_neigh, mask_prob, mask_token, num_items, rng):
        self.session = data
        self.neigh_session = neigh_s
        self.max_len_session = max_len_session
        self.max_len_neigh = max_len_neigh
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.session)

    def __getitem__(self, index):
        tokens = []
        labels = []
        # for s in self.session[index][:-1]:
        for s in self.session[index]:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(0, self.num_items))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(-1)
        # tokens.append(self.mask_token)
        # labels.append(-1)
 
        tokens = tokens[-self.max_len_session:]
        labels = labels[-self.max_len_session:]

        mask_len_s = self.max_len_session - len(tokens)
        tokens = [self.mask_token] * mask_len_s + tokens
        labels = [-1] * mask_len_s + labels

        neighs = []
        for neigh in self.neigh_session[index]:
            neigh[-1] = self.mask_token
            neigh = neigh[-self.max_len_neigh:]
            mask_len_n = self.max_len_neigh - len(neigh)
            neigh = [self.mask_token] * mask_len_n + neigh
            neighs.append(neigh)

        return torch.LongTensor(tokens), torch.LongTensor(neighs), torch.LongTensor(labels)

class BertTestDataset(Dataset):
    def __init__(self, data, neigh_s, max_len_session, max_len_neigh, mask_prob, mask_token, num_items, rng):
        self.session = data
        self.neigh_session = neigh_s
        self.max_len_session = max_len_session
        self.max_len_neigh = max_len_neigh
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.session)

    def __getitem__(self, index):
        tokens = []
        labels = []
        for s in self.session[index][:-1]:
            tokens.append(s)
            labels.append(-1)
        tokens.append(self.mask_token)
        labels.append(self.session[index][-1])
 
        tokens = tokens[-self.max_len_session:]
        labels = labels[-self.max_len_session:]

        mask_len_s = self.max_len_session - len(tokens)
        tokens = [self.mask_token] * mask_len_s + tokens
        labels = [-1] * mask_len_s + labels

        neighs = []
        for neigh in self.neigh_session[index]:
            neigh[-1] = self.mask_token
            neigh = neigh[-self.max_len_neigh:]
            mask_len_n = self.max_len_neigh - len(neigh)
            neigh = [self.mask_token] * mask_len_n + neigh
            neighs.append(neigh)

        return torch.LongTensor(tokens), torch.LongTensor(neighs), torch.LongTensor(labels)



# self.ce = nn.CrossEntropyLoss(ignore_index=0)