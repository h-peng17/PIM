
from framework import Test
from framework import Model
from framework import Config
import json 
import numpy as np 
import os 
from optparse import OptionParser
from data_loader import Data_loader


parser = OptionParser()
parser.add_option('--gpu',dest='gpu',default = 7,help = 'gpu')
parser.add_option('--mode',dest='mode',default = 'computer', help = 'mode')
parser.add_option('--ckpt_index',dest='ckpt_index',default = 1, help = 'ckpt index')
parser.add_option('--ckpt_dir', dest='ckpt_dir',default='ckpt',help='ckpt')
(options, args) =parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
p2id = json.load(open("../data/p2id.json"))
word2id = json.load(open('../data/word2id.json'))
id2word = {}
for key in word2id:
    id2word[word2id[key]] = key

config = Config()

if options.mode == 'people':
    test = Test(config, options.ckpt_dir, id2word)
    test.init_test(Model(config), options.ckpt_index)

    print('请输入:')
    line = ''
    is_continue = False
    while line != 'stop':
        line = input()
        pins = line.strip().split()
        query = np.ones([1, config.seq_len], dtype = np.int32)
        target_seq_len = [0]
        target_seq_len[0] = len(pins)
        for i, pin in enumerate(pins):
            if pin not in p2id:
                is_continue = True
                print('Invalid input!')
                break
            query[0][i] = p2id[pin]
        if is_continue:
            is_continue = False
            continue
        test.people_test_one_step(query, target_seq_len)

elif options.mode == 'computer':
    test_data_loader = Data_loader('test', config)
    test = Test(config, options.ckpt_dir, id2word, test_data_loader)
    test.init_test(Model(config))
    test._test()

    

        
    