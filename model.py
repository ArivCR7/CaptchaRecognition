import torch
from torch import nn
from torch.nn import functional as F

class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3,6), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3,6), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.linear1 = nn.Linear(4800, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.gru1 = nn.GRU(64, 32, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)
        self.output = nn.Linear(64, num_chars+1)

    def forward(self, images, targets=None):
        bs, _, _, _ = images.size()
        #print(images.shape)
        x = F.relu(self.conv1(images))
        #print(x.shape)
        x = self.maxpool1(x)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.maxpool2(x)
        #print(x.shape)
        x = x.permute(0, 3, 1, 2)
        #print(x.shape)
        x = x.view(bs, x.size(1), -1)
        #print(x.shape)
        x = F.relu(self.linear1(x))
        #print(x.shape)
        x = self.dropout1(x) 

        x, _ = self.gru1(x) # 1, 75, 64
        x = self.output(x) # 1, 75, 20
        x = x.permute(1, 0, 2) # 75, 1, 20
        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)
            input_lengths = torch.full(
                size = (bs,), fill_value=log_softmax_values.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(log_softmax_values, targets, input_lengths, target_lengths)
            return x, loss
        return x, None
