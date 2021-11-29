import torch
import torch.nn as nn
from torchvision import models
from utils.layers import Attention
from torch.nn.utils.rnn import pack_padded_sequence


class ResNet50LSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super(ResNet50LSTMAttention, self).__init__()
        encoder_net = models.resnet50(pretrained=True, progress=False)

        # Cutoff index -3 based on an output stride of 16
        modules = list(encoder_net.children())[:-3]

        self.encoder = nn.Sequential(*modules)

        enc_out_channels = 1024

        self.freeze_enc = False

        self.encoder_to_hidden = nn.Sequential(
            nn.Conv2d(enc_out_channels, hidden_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),        # batch_size, hidden_dim, 1, 1 to batch_size, hidden_dim
        )

        self.attention_block = Attention(encoder_channels=enc_out_channels, decoder_channels=hidden_dim,
                                         attention_channels=hidden_dim)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.f_beta_gate = nn.Sequential(
            nn.Linear(hidden_dim, enc_out_channels, bias=False),
            nn.GroupNorm(64, enc_out_channels),
            nn.Sigmoid(),
        )

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim + enc_out_channels, hidden_dim, num_layers=1, bidirectional=False,
                            batch_first=True, bias=True)

        self.act = nn.ReLU()
        self.d1 = nn.Dropout(p=0.3)

        self.linear = nn.Linear(hidden_dim, vocab_size, bias=True)

        self.hidden = None
        self.enc_out = None

        self.random_temp = 1.0

    def init_hidden_cell(self, enc_out):
        batch_size = enc_out.size(0)
        dtype = enc_out.dtype
        device = enc_out.device

        self.enc_out = enc_out

        initial_hidden = self.encoder_to_hidden(enc_out)

        return initial_hidden[None, ...], torch.zeros((1, batch_size, self.hidden_dim), dtype=dtype, device=device)

    def forward(self, x, captions, lengths):
        batch_size = x.size(0)

        if self.freeze_enc:
            with torch.no_grad():
                enc_out = self.encoder(x)
        else:
            enc_out = self.encoder(x)

        hidden, cell = self.init_hidden_cell(enc_out)

        embeddings = self.embedding(captions)

        if isinstance(lengths, (list, tuple)):
            lengths = torch.LongTensor(lengths)

        max_len = int(torch.max(lengths))

        outputs = torch.zeros((batch_size, max_len, self.vocab_size), dtype=enc_out.dtype).to(enc_out.device)

        # The batches are already sorted in decreasing order of lengths
        for t in range(max_len):
            batch_size_at_t = torch.sum(lengths > t)

            attended_encodings, attention_weights = self.attention_block(enc_out[:batch_size_at_t],
                                                                         hidden[-1, :batch_size_at_t])

            f_beta_gate = self.f_beta_gate(hidden[-1, :batch_size_at_t])
            f_beta_weighted_encoding = attended_encodings * f_beta_gate

            contextualized_embedding = torch.cat([embeddings[:batch_size_at_t, t], f_beta_weighted_encoding],
                                                 dim=1)

            lstm_out, (hidden, cell) = self.lstm(contextualized_embedding[:, None, :],
                                                 (hidden[:, :batch_size_at_t], cell[:, :batch_size_at_t]))

            # Remove the dummy dimension
            lstm_out = lstm_out[:, 0, :]

            lstm_out = self.d1(lstm_out)

            output = self.linear(lstm_out)

            outputs[:batch_size_at_t, t, :] = output

        packed_outputs = pack_padded_sequence(outputs, lengths, batch_first=True)

        output = packed_outputs[0]

        return output

    def get_encoder_output(self, x):
        enc_out = self.encoder(x)

        return enc_out

    def get_decoder_output(self, ind_tensor, hidden_cell):
        lstm_input = self.embedding(ind_tensor)
        if len(lstm_input.size()) == 2:
            lstm_input = lstm_input.unsqueeze(1)

        hidden, cell = hidden_cell

        attended_encodings, attention_weights = self.attention_block(self.enc_out, hidden[-1])

        f_beta_gate = self.f_beta_gate(hidden[-1])

        f_beta_weighted_encoding = attended_encodings * f_beta_gate
        f_beta_weighted_encoding = f_beta_weighted_encoding[:, None, :]

        lstm_input = torch.cat([lstm_input, f_beta_weighted_encoding], dim=2)

        lstm_out, hidden_cell = self.lstm(lstm_input, hidden_cell)

        net_output = self.linear(lstm_out)

        return net_output, hidden_cell


if __name__ == '__main__':
    # Run with dummy tensors to verify functionality
    vocab_size = 8000
    isize = 256

    net = ResNet50LSTMAttention(vocab_size=vocab_size)

    L = 20
    B = 2

    _captions = torch.randint(size=[B, L], high=vocab_size, dtype=torch.long)

    lengths = torch.randint(low=1, high=L, size=[B], dtype=torch.long)
    lengths, _ = torch.sort(lengths, descending=True)

    ip = torch.rand(size=[B, 3, isize, isize])
    print('Input size: ', ip.size(), _captions.size())

    op = net(ip, _captions, lengths)

    # Output size varies due to pack_padded_sequence
    # Output will be of size [NUM_VALID_TOKENS, vocab_size]
    print('Num valid tokens:', torch.sum(lengths))
    print('Output size:', op.size())

    print('Done')

