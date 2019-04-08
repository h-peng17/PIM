
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.pin_embedding = nn.Embedding(config.pin_size + 3, config.embedding_size, padding_idx = 1)
        self.word_embedding = nn.Embedding(config.vacab_size + 3, config.embedding_size, padding_idx = 1)
        self._init_embedding()
        self.query_seq = None 
        self.target_seq = None
    
    def _init_embedding(self):
        nn.init.xavier_normal_(self.word_embedding.weight.data) 
        self.word_embedding.weight.data[self.word_embedding.padding_idx].fill_(0)
        nn.init.xavier_normal_(self.pin_embedding.weight.data)
        self.pin_embedding.weight.data[self.pin_embedding.padding_idx].fill_(0)
    
    def forward(self):
        _sos_token = torch.zeros((self.query_seq.size()[0], 1), dtype = torch.int64).cuda()
        query_seq = self.query_seq
        target_seq = torch.cat((_sos_token, self.target_seq), 1)
        query_input = self.pin_embedding(query_seq)
        target_input = self.word_embedding(target_seq)

        return query_input, target_input


class Seq2seq(nn.Module):
    def __init__(self, config):
        super(Seq2seq, self).__init__()
        self.input_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.rnn_encoder = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size)
        self.rnn_decoder = nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vacab_size)
        self.loss = nn.CrossEntropyLoss(reduce = False)
        self._init_weight()
        self.config = config
        self.target_seq = None  # [batch_size, time_steps]
        self.loss_mask = None # [batch_size, time_steps]

    def _init_weight(self):
        # nn.init.xavier_normal_(self.rnn_encoder.weight)
        # nn.init.xavier_normal_(self.rnn_decoder.weight)
        nn.init.xavier_normal_(self.linear.weight)

    def softmax_cross_entropyloss(self, logit, target_seq, mask):
        '''
            本函数用于计算loss, 无用的loss遮蔽掉
            参数说明:
            logit: [1, batch_size, hidden_size]
            target_seq: [batch_size]
            mask: [batch_size]
        '''
        logit = self.linear(torch.squeeze(logit)) # [batch_size, vacab_size]
        loss = self.loss(logit, target_seq) #[batch_size]
        loss = torch.mean(loss * mask)
        return loss 
    
    def seq_output(self, output):
        '''
            本函数用于计算最后的输出seq
            参数说明:
            output: [batch_size, time_steps, vacab_size]
        '''
        _, output_seq = torch.max(output, dim = 2)
        return output_seq #[batch_size, time_steps]                 

    def encoder(self, input):
        '''
            本函数为seq2seq的编码部分
            参数说明:
            input [batch_size, time_steps, embedding_size]
            函数返回:
            output [time_steps, batch_size, hidden_size], 每个时间步的输出
            hidden [1, batch_size, hidden_size], 为最终的隐状态
        '''
        # [B, T, E] -> [T, B, E]
        input = input.permute(1, 0, 2)
        output, hidden = self.rnn_encoder(input)

        return output, hidden
    
    def decoder_step(self, input, hidden):
        '''
            本函数为每一个时间步的decoder
            参数说明:
            input: [batch_size, 1, embedding_size], 为当前时间步的输入
            hidden: [1, B, embedding_size], 为上一个时间步的隐状态(初始化为encoder的最后一个隐状态)
            在训练时，input为当前的预测的单词的前一个单词，第一个为<SOS>
            在测试时，input为上一个时间步的预测输出
            函数返回值:
            output: [1, batch_size, hidden_size]
            hidden: [1, batch_size, hidden_size]
        '''
        # output = F.relu(input)
        input = input.permute(1, 0, 2)
        output, hidden = self.rnn_decoder(input, hidden)
        return output, hidden
    
    def forward(self, query_input, target_input, is_training):
        '''
            本函数功能为将query_input映射为target(注意，train和eval的时候不同)
            参数说明:
            query_input: [batch_size, time_steps, embedding_size]
            target_input: [batch_size, time_steps + 1, embedding_size]
            函数返回:
            loss: 平均到每个instance的loss
        '''
        loss = 0
        _, encoder_hidden = self.encoder(query_input)
        decoder_input = torch.unsqueeze(target_input[:,0,:], 1) # init an input of [batch_size, 1, embedding_size]
        decoder_hidden = encoder_hidden
        if is_training:
            for i in range(1, target_input.size()[1]):
                decoder_output, decoder_hidden = self.decoder_step(decoder_input, decoder_hidden)
                loss += self.softmax_cross_entropyloss(decoder_output, self.target_seq[:, i-1], self.loss_mask[:, i-1])
                decoder_input = torch.unsqueeze(target_input[:, i, :], 1) 
            return loss
        else:
            output = [] # list of [1, batch_size, hidden_size]
            for i in range(1, target_input.size()[1]):
                decoder_output, decoder_hidden = self.decoder_step(decoder_input, decoder_hidden)
                decoder_input = decoder_output
                output.append(decoder_output)
            # [time_steps, batch_size, vacab_size] -> [batch_size, time_steps, vacab_size]
            output = self.linear(torch.cat(output, 0)).permute(1, 0, 2)
            return self.seq_output(output)

    

