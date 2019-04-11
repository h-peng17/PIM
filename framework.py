
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
        loss, output = self.seq2seq(query_input, target_input, True)
        return loss, output 
    
    def test(self):
        query_input, target_input = self.embedding()
        output = self.seq2seq(query_input, target_input, False)
        return output #[batch_size, time_steps]

class Config():
    def __init__(self):
        self.vacab_size = 6763
        self.pin_size = 406
        self.batch_size = 128
        self.lr = 0.001 
        self.max_epoch = 1000
        self.embedding_size = 300
        self.seq_len = 20
        self.hidden_size = 1024
        self.weight_decay = 1e-5
        self.save_epoch = 1
        self.model_name = 'seq2seq'
        self.loss_save = 1000

    
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
        self.correct = 0
        self.all = 0

    def init_train(self, model):
        self.train_model = model
        # device_ids = [0,1,2]
        self.train_model.cuda()
        # self.train_model = nn.DataParallel(self.train_model, device_ids = device_ids)
        self.train_model.train()
    
        parameters_to_optimize = filter(lambda x: x.requires_grad, self.train_model.parameters())
        self.optimizer = optim.Adam(parameters_to_optimize, lr = self.config.lr, weight_decay = self.config.weight_decay)

    def to_var(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()
    
    def train_one_step(self):
        batch = self.train_data_loader.next_batch()
        if batch == None:
            batch = self.train_data_loader.next_batch()
        self.train_model.embedding.query_seq = self.to_var(batch["query_seq"])
        self.train_model.embedding.target_seq = self.to_var(batch["target_seq"])
        self.train_model.seq2seq.target_seq = self.to_var(batch["target_seq"])
        self.train_model.seq2seq.loss_mask = torch.from_numpy(batch["query_mask"]).to(torch.float32).cuda()

        target_seq = np.multiply(batch["target_seq"], batch["query_mask"]) #[batch, time_steps]
        self.all += np.sum(batch["target_seq_len"]) # total

        self.optimizer.zero_grad()
        loss, output = self.train_model()
        loss.backward()
        self.optimizer.step()

        output = np.array(((output.cpu()).detach()))
        output = np.multiply(output, batch["query_mask"])
        self.correct += np.logical_and(output == target_seq, output != 0).sum()
        accuracy = self.correct / self.all

        return loss, accuracy 
    
    def _train(self):
        print("begin training...")
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        train_order = self.train_data_loader.order 
        for epoch in range(self.config.max_epoch):
            for i in range(int(len(train_order) / self.config.batch_size)):
                loss, accuracy = self.train_one_step()
                sys.stdout.write('epoch:{}, batch:{}, accuracy:{}, loss:{}\r'.format(epoch, i, round(accuracy, 6), loss))
                sys.stdout.flush()


            if (epoch + 1) % self.config.save_epoch == 0:
                print('epoch:{} has saved'.format(epoch))
                path = os.path.join(self.ckpt_dir, self.config.model_name + '-' + str(epoch))
                torch.save(self.train_model.state_dict(), path)
            
                
class Test():
    def __init__(self, config, ckpt_dir, id2word):
        self.config = config 
        self.ckpt_dir = ckpt_dir
        self.id2word = id2word

    def init_test(self, model, ckpt_index):
        print('init test model....')
        self.test_model = model
        self.test_model.cuda()
        self.test_model.eval()
        path = os.path.join(self.ckpt_dir, self.config.model_name + '-' + str(ckpt_index))
        self.test_model.load_state_dict(torch.load(path))
    
    def to_val(self, x):
        return torch.from_numpy(x).to(torch.int64).cuda()
    
    def target_seq(self, batch_size):
        return torch.zeros((batch_size, self.config.seq_len)).to(torch.int64).cuda()
    
    def convert2word(self, output, target_seq_len):
        '''
            output: [1, seq_len]
            target_seq_len: [1]
        '''
        out_file = ''
        for i in range(len(target_seq_len)):
            seq = ''
            for j in range(target_seq_len[i]):
                seq += self.id2word[output[i][j]] + ' '
            seq += '\r'
            out_file += seq
        print(out_file)
        with open('../data/output.txt', 'w', encoding = 'utf8') as f:
            f.write(out_file)
        # print('done!')

    def test_one_step(self, query_seq, target_seq_len):
        self.test_model.embedding.query_seq = self.to_val(query_seq)
        self.test_model.embedding.target_seq = self.target_seq(1)

        output = self.test_model.test() #[1, seq_len]
        output = np.array((output.cpu()).detach())
        output = output.tolist()
        self.convert2word(output, target_seq_len)
    




        