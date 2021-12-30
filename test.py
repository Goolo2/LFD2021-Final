import numpy as np
import random
import torch


# k = np.array([1, 2, 3])
# x = torch.from_numpy(state['ope_seq'].astype(np.int64)).cuda(device)

if (torch.cuda.is_available()):
    print('fuck')