import numpy as np
import IPython
from scipy.special import softmax

class Sampler(object):
    def __init__(self, sample_num=50000):
        self.sample_num=sample_num
        self.rs = np.random.RandomState(1234)

    def sample_full(self, dist, anno_mat):
        sampled_anno = anno_mat.copy()
        sampled_prob = np.zeros_like(anno_mat, dtype=np.float)
        for i in range(anno_mat.shape[0]):
            for j in range(anno_mat.shape[1]):
                a = dist[i][j].copy().astype(float)
                a = a / a.sum()
                try:
                    anno = np.argmax(self.rs.multinomial(1, a))
                except:
                    IPython.embed()
                sampled_anno[i][j] = anno
                sampled_prob[i][j] = a[anno]
        return sampled_prob, sampled_anno

    def select_wrt_entropy(self, anno_mat, sampled_mat, entropy, ratio=1., reverse=True):
        sampled_anno = anno_mat.copy()
        for j in range(anno_mat.shape[1]):
            num_nonety = np.sum(anno_mat[:, j] != -1)
            ety_idx = np.arange(anno_mat.shape[0])[anno_mat[:, j] == -1]

            if reverse:
                ety_dist = 1 / entropy[ety_idx][:, j] 
            else:
                ety_dist = entropy[ety_idx][:, j]
            ety_dist = ety_dist / ety_dist.sum()
            try:
                selected_idx = self.rs.choice(len(ety_idx), int(num_nonety * ratio) if int(num_nonety * ratio) < len(ety_idx) - np.sum(ety_dist == 0) else len(ety_idx) - np.sum(ety_dist == 0),
                                          replace=False, p=ety_dist)
            except:
                IPython.embed()
            sampled_anno[:, j][ety_idx[selected_idx]] = sampled_mat[:, j][ety_idx[selected_idx]]
        return sampled_anno

