import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def fix_random_seed_as(random_seed=2022):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False