import torch
import torch.nn as nn
from torchvision import models
from torch.nn.utils.rnn import pack_padded_sequence


class ResNet50LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, use_relu=True, embeddings=None):
        super(ResNet50LSTM, self).__init__()
        encoder_net = models.resnet50(pretrained=True, progress=False)

        modules = list(encoder_net.children())[:-1]

        self.encoder = nn.Sequential(*modules)

        self.encoder_fc = nn.Sequential(
            nn.Linear(encoder_net.fc.in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

        self.freeze_enc = False

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        if embeddings is not None:
            v_size, e_dim = embeddings.size()

            assert v_size == vocab_size

            if e_dim > embedding_dim:
                embeddings = embeddings[:, :embedding_dim]
            elif e_dim < embedding_dim:
                min_val = torch.min(embeddings)
                max_val = torch.max(embeddings)
                new_embeddings = min_val + (torch.rand(size=[v_size, embedding_dim]) * (max_val - min_val))
                new_embeddings[:, :e_dim] = embeddings
                embeddings = new_embeddings

        self.embedding = nn.Embedding(vocab_size, embedding_dim, _weight=embeddings)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=True,
                            bias=True)

        self.act = nn.ReLU()
        self.use_relu = use_relu

        self.d1 = nn.Dropout(p=0.2)

        self.linear = nn.Linear(hidden_dim, vocab_size, bias=True)

        self.hidden = None

        self.random_temp = 1.0

    def init_hidden_cell(self, enc_out):
        batch_size = enc_out.size(0)
        dtype = enc_out.dtype
        device = enc_out.device

        return enc_out[None, ...], torch.zeros((1, batch_size, self.hidden_dim), dtype=dtype, device=device)

    def forward(self, x, captions, lengths):
        if self.freeze_enc:
            with torch.no_grad():
                enc_out = self.encoder(x)
        else:
            enc_out = self.encoder(x)

        enc_out = self.encoder_fc(torch.flatten(enc_out, 1))

        self.hidden = self.init_hidden_cell(enc_out)

        captions = captions[:, :-1]

        embeddings = self.embedding(captions)

        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, self.hidden = self.lstm(packed, self.hidden)
        lstm_out = lstm_out[0]

        lstm_out = self.d1(lstm_out)
        if self.use_relu:
            lstm_out = self.act(lstm_out)

        output = self.linear(lstm_out)

        return output

    def get_encoder_output(self, x):
        enc_out = self.encoder(x)
        enc_out = self.encoder_fc(torch.flatten(enc_out, 1))

        return enc_out

    def get_decoder_output(self, ind_tensor, hidden_cell):
        lstm_input = self.embedding(ind_tensor)
        if len(lstm_input.size()) == 2:
            lstm_input = lstm_input.unsqueeze(1)

        lstm_out, hidden_cell = self.lstm(lstm_input, hidden_cell)
        if self.use_relu:
            lstm_out = self.act(lstm_out)
        net_output = self.linear(lstm_out)

        return net_output, hidden_cell


if __name__ == '__main__':
    # Run with dummy tensors to verify functionality
    vocab_size = 8000

    net = ResNet50LSTM(vocab_size=vocab_size)

    L = 20
    B = 2

    _captions = torch.randint(size=[B, L], high=vocab_size, dtype=torch.long)

    lengths = torch.randint(low=1, high=L, size=[B], dtype=torch.long)
    lengths, _ = torch.sort(lengths, descending=True)

    w = 256
    h = 256

    ip = torch.rand(size=[B, 3, h, w])
    print('Input size: ', ip.size(), _captions.size())

    op = net(ip, _captions, lengths)

    # Output size varies due to pack_padded_sequence
    # Output will be of size [NUM_VALID_TOKENS, vocab_size]
    print('Output size:', op.size())

    print('Done')

