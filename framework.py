
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np 
from data_loader import Data_loader
from seq2seq import Embedding
from seq2seq import Seq2seq
import os 
import sys 

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.embedding = Embedding(config)
        self.seq2seq = Seq2seq(config)

    def forward(self):
        query_input, target_input = self.embedding()
        loss = self.seq2seq(query_input, target_input, True)
        return loss 
    
    def test(self):
        query_input, target_input = self.embedding()
        output = self.seq2seq(query_input, target_input, False)
        return output #[batch_size, time_steps]

class Config():
    def __init__(self):
        self.vacab_size = 6763
        self.pin_size = 406
        self.batch_size = 160 
        self.lr = 0.5 
        self.max_epoch = 60
        self.embedding_size = 50
        self.seq_len = 20
        self.hidden_size = 230
        self.weight_decay = 0
        self.save_epoch = 2
        self.model_name = 'seq2seq'

    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_lr(self, lr):
        self.lr = lr
    
    def set_max_epoch(self, max_epoch):
        self.max_epoch = max_epoch
    
    def set_embedding_size(self, embedding_size):
        self.embedding_size = embedding_size
    
    def set_hidden_size(self, hidden_size):
        self.hidden_size = hidden_size
    
    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay
    

class Train():
    def __init__(self, train_data_loader, config, ckpt_dir):
        self.config = config
        self.train_data_loader = train_data_loader
        self.ckpt_dir = ckpt_dir

    def init_train(self, model):
        self.train_model = model
        self.train_model.cuda()
        self.train_model.train()
    
        parameters_to_optimize = filter(lambda x: x.requires_grad, self.train_model.parameters())
        self.optimizer = optim.SGD(parameters_to_optimize, lr = self.config.lr, weight_decay = self.config.weight_decay)

    def to_var(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()
    
    def train_one_step(self):
        batch = self.train_data_loader.next_batch()
        self.train_model.embedding.query_seq = self.to_var(batch["query_seq"])
        self.train_model.embedding.target_seq = self.to_var(batch["target_seq"])
        self.train_model.seq2seq.target_seq = self.to_var(batch["target_seq"])
        self.train_model.seq2seq.loss_mask = torch.from_numpy(batch["query_mask"]).to(torch.float32).cuda()

        self.optimizer.zero_grad()
        loss = self.train_model()
        loss.backward()
        self.optimizer.step()

        return loss 
    
    def _train(self):
        print("begin training...")
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        
        train_order = self.train_data_loader.order 
        for epoch in range(self.config.max_epoch):
            for i in range(int(len(train_order) / self.config.batch_size)):
                loss = self.train_one_step()
                sys.stdout.write('epoch:{}, batch:{}, loss:{}\r'.format(epoch, i, loss))
                sys.stdout.flush()

            if (epoch + 1) % self.config.save_epoch == 0:
                print('epoch:{} has saved'.format(epoch))
                path = os.path.join(self.ckpt_dir, self.config.model_name + '-' + str(epoch))
                torch.save(self.train_model.state_dict(), path)
                
class Test():
    def __init__(self, test_data_loader, config, ckpt_dir, id2word):
        self.test_data_loader = test_data_loader
        self.config = config 
        self.ckpt_dir = ckpt_dir
        self.id2word = id2word

    def init_test(self, model):
        print('init test model....')
        self.test_model = model
        self.test_model.cuda()
        self.test_model.eval()
    
    def to_val(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()
    
    def target_seq(self, batch_size):
        return torch.zeros((batch_size, self.config.seq_len)).to(torch.int64).cuda()
    
    def convert2word(self, output, target_seq_len):
        '''
            output: [batch, seq_len]
            target_seq_len: [batch]
        '''
        out_file = ''
        for i in range(len(target_seq_len)):
            seq = ''
            for j in range(target_seq_len[i]):
                seq += self.id2word[output[i][j]] + ' '
            seq += '\r'
            out_file += seq
        with open('output.txt', 'w') as f:
            f.write(out_file)
        print('done!')

    def test_one_step(self):
        batch = self.test_data_loader.next_batch()
        self.test_model.embedding.query_seq = self.to_val(batch["query_seq"])
        self.test_model.embedding.target_seq = self.target_seq(batch["query_seq"].shape()[0])
        target_seq_len = batch["target_seq_len"].tolist()

        output = self.test_model.test() #[batch, seq_len]
        output = np.array((output.cpu()).detach())
        output = output.tolist()
        self.convert2word(output, target_seq_len)
    

config = Config()
train_data_loader = Data_loader('train', config)
train = Train(train_data_loader, config, '../ckpt')
train.init_train(Model(config))
train._train()


        