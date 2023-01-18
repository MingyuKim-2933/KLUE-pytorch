import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, weights, input_size, hidden_size, num_layers, num_classes, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        # random embedding
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    # def from_pretrained(self, weights):
    #     self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)
    #     print("Successfully load the pre-trained embeddings.")

    def forward(self, x):
        x = self.embedding(x)

        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate rnn
        out, _ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out