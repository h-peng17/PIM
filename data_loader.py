
import numpy as np 
import random

class Data_loader():
    def __init__(self, mode, config):
        '''
            query_seq: [total, time_steps]
            query_mask: [total, time_steps]
            taret_seq: [total, time_steps]
            target_seq_len: [total]
        '''
        self.query_seq = np.load("../data/{}_query_seq.npy".format(mode)) 
        self.query_mask = np.load("../data/{}_query_mask.npy".format(mode))
        self.target_seq = np.load("../data/{}_target_seq.npy".format(mode))
        self.target_seq_len = np.load("../data/{}_target_seq_len.npy".format(mode))

        self.config = config
        self.idx = 0
        self.order =list(range(len(self.query_seq)))
        self.instance_total = len(self.query_seq)

    def next_batch(self):
        if self.idx >= self.instance_total:
            self.idx = 0
            random.shuffle(self.order)
        id0 = self.idx 
        id1 = self.idx + self.config.batch_size
        self.idx = id1 
        if id1 >= self.instance_total:
            # id1 = self.instance_total
            return None
        index = self.order[id0:id1]
        batch = {}
        batch["query_seq"] = self.query_seq[index]
        batch["query_mask"] = self.query_mask[index]
        batch["target_seq"] = self.target_seq[index]
        batch["target_seq_len"] = self.target_seq_len[index]
    
        return batch