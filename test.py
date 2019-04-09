
from framework import Test
from framework import Model
from framework import Config
import json 
import numpy as np 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
p2id = json.load(open("../data/p2id.json"))
word2id = json.load(open('../data/word2id.json'))
id2word = {}
for key in word2id:
    id2word[word2id[key]] = key

config = Config()
test = Test(config, '../ckpt', id2word)
test.init_test(Model(config))

line = ''
while line != 'stop':
    line = input()
    pins = line.strip().split()
    query = np.ones([1, config.seq_len], dtype = np.int32)
    target_seq_len = [0]
    target_seq_len[0] = len(pins)
    for i, pin in enumerate(pins):
        query[0][i] = p2id[pin]
    test.test_one_step(query, target_seq_len)

    

        
    