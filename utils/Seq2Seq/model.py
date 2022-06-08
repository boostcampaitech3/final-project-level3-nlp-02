import torch.nn as nn
import torch
import random


# decoder
class Decoder(nn.Module):
    # Decoder(trg_vocab_size, emb_dim, 256, lstm_layer, drop_rate)
    def __init__(self, output_dim, input_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers


        self.rnn = torch.nn.LSTM(input_size=input_dim,
                            hidden_size=hid_dim,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=dropout,
                            )

        self.fc_out = nn.Linear(hid_dim*n_layers, 32002)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        
        return output, hidden, cell

class BERT_LSTM(torch.nn.Module):
    def __init__(self, backbone, tokenizer, hidden_size=768, decoder_hidden=512 ,drop_rate=0.3, lstm_layer=2, batch_size=32, trg_vocab_size=32002):
        super(BERT_LSTM, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=tokenizer
        self.Backbone=backbone
        self.batch_size=batch_size
        self.trg_vocab_size=trg_vocab_size
        self.hdim=hidden_size
        self.n_layer=lstm_layer
        self.decoder_hidden = decoder_hidden

        self.LSTM = Decoder(trg_vocab_size, hidden_size, decoder_hidden, lstm_layer, drop_rate)
        self.tanh=torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(p=drop_rate)
        self.fc1_out = nn.Linear(decoder_hidden*2, 32002)
        self.fc2_out = nn.Linear(hidden_size, 32002)


    def forward(self, input, teacher_forcing_ratio=1.0):

        batch_size=input['input_ids'].shape[0]

        # print("입력 시퀀스",input['input_ids'].shape)
        # print("출력 시퀀스",input['label_ids'].shape)
        bert_out = self.Backbone(input_ids=input['input_ids'].to(self.device),attention_mask=input['attention_mask'].to(self.device),token_type_ids=input['token_type_ids'].to(self.device))
        # print("Encoder hidden vector", bert_out['last_hidden_state'].shape)

        # ([64, 42, 768]) -> ([4, 64, 256]) : dim 1 기준으로 더하고, [64,768]로 만든걸 [4,64,256] 으로 변환해야함
        # 그냥 LSTM 인코더 입력 차원을 768로 쓰고, layer개수 1개 일반 lstm으로 해서 차원 맞춰서 돌려보자
        # 최종적으로 [64,42,768] -> [64,768] (합하고 길이로 나누고, 2번 쌓기) -> [2,64,768]
        Encoder_hidden = bert_out['last_hidden_state']
        Encoder_hidden = torch.sum(Encoder_hidden,axis=1)/bert_out['last_hidden_state'].size(1)
        Encoder_hidden =  torch.stack([Encoder_hidden,Encoder_hidden], dim=0) 

        input_token = bert_out['last_hidden_state'][:,0,:].unsqueeze(1)

        trg_len=input['label_ids'].size(1)

        # decoder의 output을 저장하기 위한 tensor
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size).to(self.device)

        hidden = Encoder_hidden
        cell = Encoder_hidden


        #### TODO outputs 저장할 때, LSTM output이랑 bert_out 평균내서 저장하자!! (그럼 bert_out를 fc로 변환해서 넣어줘야겠네)
        for t in range(1,trg_len): # <eos> 제외하고 trg_len-1 만큼 반복
            output, hidden, cell = self.LSTM(input_token, hidden, cell)

            input_token = bert_out['last_hidden_state'][:,t,:].unsqueeze(1)

            output = (self.fc1_out(output.squeeze(1)).to(self.device) + self.fc2_out(input_token.squeeze(1)).to(self.device))/2

            # prediction 저장
            outputs[:,t,:] = output

            

        return outputs