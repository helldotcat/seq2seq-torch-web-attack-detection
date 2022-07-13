import random

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, num_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 4 * num_layers, dec_hid_dim * 2 * num_layers)
        self.dropout = nn.Dropout(dropout)

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.num_layers = num_layers
        
    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        _, hidden = self.rnn(embedded)
        # hidden = ([n layers * num directions, batch size, hid dim],
        #           [n layers * num directions, batch size, hid dim])

        hidden = torch.cat(
            (
                hidden[0].transpose(1, 0).reshape(-1, self.enc_hid_dim * self.num_layers * 2),
                hidden[1].transpose(1, 0).reshape(-1, self.enc_hid_dim * self.num_layers * 2)
            ),
            dim=1
        )
        # hidden = [batch size, enc hid dim * num_layers * num directions * 2 (hidden and cell)]

        # pass hidden state through the FC layer 
        hidden = torch.tanh(self.fc(hidden))
        # hidden = [batch size, dec hid dim * num_layers * 2 (hidden and cell)]

        hidden = hidden.reshape(-1, self.num_layers * 2, self.dec_hid_dim).permute(1, 0, 2)
        # hidden = [num_layers * 2 (hidden and cell), batch size, dec hid dim]

        hidden = (
            hidden[:self.num_layers, :].contiguous(),
            hidden[self.num_layers:, :].contiguous()
        )

        # hidden = ([num_layers, batch size, dec hid dim], [num_layers, batch size, dec hid dim])
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, dec_hid_dim, num_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, num_layers=num_layers, dropout=dropout)
        self.fc_out = nn.Linear(dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):   
        # input = [batch size]
        # hidden = ([num layers, batch size, dec hid dim], [num layers, batch size, dec hid dim])

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        output, hidden = self.rnn(embedded, hidden)
        # output = [1, batch size, dec hid dim]
        # hidden = ([num layers, batch size, dec hid dim], [num layers, batch size, dec hid dim])

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        # embedded = [batch size, emb dim]
        # output = [batch size, dec hid dim]
        
        prediction = self.fc_out(torch.cat((output, embedded), dim=1))
        # prediction = [batch size, output dim]

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src: torch.Tensor, teacher_forcing_ratio: float):
        # src = [src len, batch size]

        batch_size = src.shape[1]
        max_len = src.shape[0]
        vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        hidden = self.encoder(src)
        # hidden = ([num_layers, batch size, dec hid dim], [num_layers, batch size, dec hid dim])

        # first input to the decoder is the <sos> tokens
        input = src[0, :]
        # input = [batch size]

        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            # output = [batch size, vocab size]
            # hidden = ([num_layers, batch size, dec hid dim], [num_layers, batch size, dec hid dim])

            outputs[t] = output

            input = src[t] if random.random() <= teacher_forcing_ratio else output.max(1)[1]

        return outputs


def model_init(model):
    if isinstance(model, nn.Linear):
        nn.init.xavier_normal_(model.weight.data)
        nn.init.normal_(model.bias.data)

    elif isinstance(model, nn.LSTM):
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param.data)
            else:
                nn.init.normal_(param.data)
