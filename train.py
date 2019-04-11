
from framework import Train
from framework import Model
from framework import Config
from data_loader import Data_loader
import os 
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--gpu',dest='gpu',default = 7,help = 'gpu')
parser.add_option('--ckpt_index',dest='ckpt_index',default = 1, help = 'ckpt index')
parser.add_option('--ckpt_dir', dest='ckpt_dir',default='ckpt',help='ckpt')
(options, args) =parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu
config = Config()
train_data_loader = Data_loader('train', config)
train = Train(train_data_loader, config, options.ckpt_dir)
train.init_train(Model(config))
train._train()