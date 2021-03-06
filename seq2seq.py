
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
        self.rnn_encoder = nn.GRU(input_size = self.hidden_size, hidden_size = self.hidden_size, num_layers = 4)
        self.rnn_decoder = nn.GRU(input_size = self.hidden_size, hidden_size = self.hidden_size, num_layers = 4)
        self.linear = nn.Linear(config.hidden_size * 2, config.vacab_size + 3)
        self.embed2input = nn.Linear(config.embedding_size, config.hidden_size)
        self.loss = nn.CrossEntropyLoss(reduce = False)
        self._init_weight()
        self.config = config
        self.dropout = nn.Dropout(0.5)
        self.target_seq = None  # [batch_size, time_steps]
        self.loss_mask = None # [batch_size, time_steps]
        

    def _init_weight(self):
        # nn.init.xavier_normal_(self.rnn_encoder.weight)
        # nn.init.xavier_normal_(self.rnn_decoder.weight)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.embed2input.weight)

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
            input [batch_size, time_steps, hidden_size]
            函数返回:
            output [time_steps, batch_size, hidden_size], 每个时间步的输出
            hidden [4, batch_size, hidden_size], 为最终的隐状态
        '''
        # [B, T, E] -> [T, B, E]
        input = input.permute(1, 0, 2)
        output, hidden = self.rnn_encoder(input)

        return output, hidden
    
    def decoder_step(self, input, hidden):
        '''
            本函数为每一个时间步的decoder
            参数说明:
            input: [1, B, hidden_size], 为当前时间步的输入
            hidden: [4, B, hidden_size], 为上一个时间步的隐状态(初始化为encoder的最后一个隐状态)
            在训练时，input为当前的预测的单词的前一个单词，第一个为<SOS>
            在测试时，input为上一个时间步的预测输出
            函数返回值:
            output: [1, batch_size, hidden_size]
            hidden: [4, batch_size, hidden_size]
        '''
        # output = F.relu(input)
        # input = input.permute(1, 0, 2)
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
        # 首先做一下映射
        self.dropout(query_input)
        self.dropout(target_input)
        query_input = self.embed2input(query_input)
        target_input = self.embed2input(target_input)
        loss = 0
        encoder_output, encoder_hidden = self.encoder(query_input)
        # self.dropout(encoder_output)
        # self.dropout(encoder_hidden)
        encoder_output = encoder_output.permute(1, 0, 2) # [batch, time_step, hidden_size]
        decoder_input = torch.unsqueeze(target_input[:,0,:], 0) # init an input of [1, batch_size, hidden_size]
        decoder_hidden = encoder_hidden
        if is_training:
            output = []
            for i in range(1, target_input.size()[1]):
                decoder_output, decoder_hidden = self.decoder_step(decoder_input, decoder_hidden)
                
                decoder_output = decoder_output.permute(1, 2, 0) #[batch, hidden_size, 1]
                weight = (torch.bmm(encoder_output, decoder_output)).permute(0, 2, 1) #[batch, time_step, 1] -> [batch, 1, timesteps]
                context = (torch.bmm(weight, encoder_output)).permute(1, 0, 2) #[1, batch_size, hidden_size]
                decoder_output = torch.cat((context, decoder_output.permute(2, 0, 1)), 2) #[1, batch_size, hidden_size * 2]
                output.append(decoder_output) # list of [1, batch, hidden_size * 2] 

                loss += self.softmax_cross_entropyloss(decoder_output, self.target_seq[:, i-1], self.loss_mask[:, i-1])
                decoder_input = torch.unsqueeze(target_input[:, i, :], 0) 

            # [time_steps, batch_size, vacab_size] -> [batch_size, time_steps, vacab_size]
            output = self.linear(torch.cat(output, 0)).permute(1, 0, 2)
            # loss_mask = self.loss_mask.to(torch.float32)
            return loss, self.seq_output(output) # [batch_size, time_steps]
        else:
            output = [] # list of [1, batch_size, hidden_size * 2]
            for i in range(1, target_input.size()[1]):
                decoder_output, decoder_hidden = self.decoder_step(decoder_input, decoder_hidden)
                decoder_input = decoder_output
                # decode_output [1, batch, hidden_size] -> [batch, hidden_size, 1]
                # encoder_output [batch, time_steps, hidden_size]
                decoder_output = decoder_output.permute(1, 2, 0)
                weight = (torch.bmm(encoder_output, decoder_output)).permute(0, 2, 1) #[batch, time_step, 1] -> [batch, 1, timesteps]
                context = (torch.bmm(weight, encoder_output)).permute(1, 0, 2) #[batch, 1, hidden_size] -> [1, batch, hidden_size]
                
                output.append(torch.cat((context, decoder_input), 2))
            # [time_steps, batch_size, vacab_size] -> [batch_size, time_steps, vacab_size]
            output = self.linear(torch.cat(output, 0)).permute(1, 0, 2)
            return self.seq_output(output)




