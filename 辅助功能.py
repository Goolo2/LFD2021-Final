import numpy as np


def combine_states(img_tensor, ope_seq, trg_mask):
    state = {}
    state['img_tensor'] = img_tensor[np.newaxis, :]
    state['ope_seq'] = ope_seq
    state['trg_mask'] = trg_mask
    return state
