
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

'''
CNN and CRNN classes, loss and metrics.
'''

def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

def loss_fn(prediction, target):
    loss = nn.MSELoss()
    output = loss(prediction, target)
    return output


class CNN(nn.Module):

    def __init__(self, channels):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)    # output: (batch,96,192,32)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, 4, 2, 1)  #  (batch,48,96,64)
        self.conv2_bn = nn.BatchNorm2d(channels*2)
        self.conv3 = nn.Conv2d(channels*2, channels*4, 4, 2, 1) # (batch,24,48,128)
        self.conv3_bn = nn.BatchNorm2d(channels*4)
        self.conv4 = nn.Conv2d(channels*4, channels*8, 4, 2, 1) # (batch,12,24,256)
        self.conv4_bn = nn.BatchNorm2d(channels*8)
        self.conv5 = nn.Conv2d(channels*8, channels*16, 4, 2, 1) # (batch,6,12,512)
        self.conv5_bn = nn.BatchNorm2d(channels*16)
        self.conv6 = nn.Conv2d(channels*16, channels*16, 4, 1, 0) # (batch,3,6,1024)
        self.fc = nn.Linear(channels*16*3*9, 1)

    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))

        # resize tensor to fit the FC
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x.view(-1)



class CRNN(nn.Module):

    def __init__(self, channels, vector_dim, rnn_hidden_size, rnn_num_layers):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)    # output: (batch,96,192,32)
        self.conv1_bn = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels*2, 4, 2, 1)  #  (batch,48,96,64)
        #self.conv2_drop = nn.Dropout2d(p=0.5)
        self.conv2_bn = nn.BatchNorm2d(channels*2)
        self.conv3 = nn.Conv2d(channels*2, channels*4, 4, 2, 1) # (batch,24,48,128)
        #self.conv3_drop = nn.Dropout2d(p=0.5)
        self.conv3_bn = nn.BatchNorm2d(channels*4)
        self.conv4 = nn.Conv2d(channels*4, channels*8, 4, 2, 1) # (batch,12,24,256)
        #self.conv4_drop = nn.Dropout2d(p=0.5)
        self.conv4_bn = nn.BatchNorm2d(channels*8)
        self.conv5 = nn.Conv2d(channels*8, channels*16, 4, 2, 1) # (batch,6,12,512)
        self.conv5_bn = nn.BatchNorm2d(channels*16)
        self.conv6 = nn.Conv2d(channels*16, channels*16, 4, 1, 0) # (batch,3,6,1024)
        #self.conv6_drop = nn.Dropout2d(p=0.5)

        # Linear
        self.fc1 = nn.Linear(channels*16*3*9, vector_dim)
        #self.fc1_drop = nn.Dropout(p=0.5)

        # RNN
        self.hidden_size = rnn_hidden_size
        self.num_layers = rnn_num_layers
        self.lstm = nn.LSTM(input_size=vector_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)

        # Linear
        self.fc2 = nn.Linear(self.hidden_size, 1)


    def forward(self, input):

        batch_size, shape1, T, H, W = input.shape
        input = input.reshape(batch_size*T, shape1, H, W)
        #h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # CNN
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6(x))

        # resize tensor and FC
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = x.reshape(batch_size, T, x.shape[-1])

        # RNN
        out, _ = self.lstm(x, (h0, c0))

        # Decode hidden state of last time step
        out = self.fc2(out[:, -1, :])

        return out.view(-1) #lstm1_out

'''
Here we can add the metrics we decide to use.
'''
