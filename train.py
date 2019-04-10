
from framework import Train
from framework import Model
from framework import Config
from data_loader import Data_loader
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

config = Config()
train_data_loader = Data_loader('train', config)
train = Train(train_data_loader, config, '../ckpt_1')
train.init_train(Model(config))
train._train()