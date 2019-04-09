
from framework import Train
from framework import Model
from framework import Config
from data_loader import Data_loader
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = Config()
train_data_loader = Data_loader('train', config)
train = Train(train_data_loader, config, '../ckpt')
train.init_train(Model(config))
train._train()