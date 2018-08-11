
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision.models as models

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

    def __init__(self, variables, channels, vector_dim, dropout=None):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CNN, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(variables, channels, 3, stride=1, padding=1)    # output: (batch,96,192,32)
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
        # Linear
        self.fc1 = nn.Linear(channels*16*3*9, vector_dim)
        # dropout regularization
        self.dropout = dropout
        if dropout != None:
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.conv3_drop = nn.Dropout2d(p=dropout)
            self.conv4_drop = nn.Dropout2d(p=dropout)
            self.conv5_drop = nn.Dropout2d(p=dropout)
            self.conv6_drop = nn.Dropout2d(p=dropout)
            self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, input):

        if self.dropout == None:
            x = F.relu(self.conv1_bn(self.conv1(input)))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4(x)))
            x = F.relu(self.conv5_bn(self.conv5(x)))
            x = F.relu(self.conv6(x))
            # resize tensor and FC
            x = x.view(x.size()[0], -1)
            x = self.fc1(x)
        else:
            x = F.relu(self.conv1_bn(self.conv1(input)))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4_drop(self.conv4(x))))
            x = F.relu(self.conv5_bn(self.conv5_drop(self.conv5(x))))
            x = F.relu(self.conv6_drop(self.conv6(x)))
            # resize tensor and FC
            x = x.view(x.size()[0], -1)
            x = self.fc1_drop(self.fc1(x))
        return x
        #eturn x.view(-1)


class RNN(nn.Module):

    def __init__(self, vector_dim, rnn_hidden_size, rnn_num_layers):
        super(RNN, self).__init__()
        self.hidden_size = rnn_hidden_size
        self.num_layers = rnn_num_layers
        self.lstm = nn.LSTM(input_size=vector_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # Linear
        self.fc2 = nn.Linear(self.hidden_size, 1)

    def forward(self, input):

        batch_size, T, dim = input.shape
        if torch.cuda.is_available():
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        out, _ = self.lstm(input, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc2(out[:, -1, :])
        return out

class CRNN(nn.Module):

    def __init__(self, variables, channels, vector_dim, rnn_hidden_size, rnn_num_layers, dropout=None):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(variables, channels, 3, stride=1, padding=1)    # output: (batch,96,192,32)
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
        # Linear
        self.fc1 = nn.Linear(channels*16*3*9, vector_dim)
        # dropout regularization
        self.dropout = dropout
        if dropout != None:
            self.conv2_drop = nn.Dropout2d(p=dropout)
            self.conv3_drop = nn.Dropout2d(p=dropout)
            self.conv4_drop = nn.Dropout2d(p=dropout)
            self.conv5_drop = nn.Dropout2d(p=dropout)
            self.conv6_drop = nn.Dropout2d(p=dropout)
            self.fc1_drop = nn.Dropout(p=dropout)

        self.hidden_size = rnn_hidden_size
        self.num_layers = rnn_num_layers
        self.lstm = nn.LSTM(input_size=vector_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        # Linear
        self.fc2 = nn.Linear(self.hidden_size, 1)


    def forward(self, input):

        # CNN
        batch_size, shape1, T, H, W = input.shape
        if torch.cuda.is_available():
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
        else:
            h0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        input = input.reshape(batch_size*T, shape1, H, W)
        if self.dropout == None:
            x = F.relu(self.conv1_bn(self.conv1(input)))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4(x)))
            x = F.relu(self.conv5_bn(self.conv5(x)))
            x = F.relu(self.conv6(x))
            # resize tensor and FC
            x = x.view(x.size()[0], -1)
            x = self.fc1(x)
        else:
            x = F.relu(self.conv1_bn(self.conv1(input)))
            x = F.relu(self.conv2_bn(self.conv2(x)))
            x = F.relu(self.conv3_bn(self.conv3(x)))
            x = F.relu(self.conv4_bn(self.conv4_drop(self.conv4(x))))
            x = F.relu(self.conv5_bn(self.conv5_drop(self.conv5(x))))
            x = F.relu(self.conv6_drop(self.conv6(x)))
            # resize tensor and FC
            x = x.view(x.size()[0], -1)
            x = self.fc1_drop(self.fc1(x))

        x = x.reshape(batch_size, T, x.shape[-1])

        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc2(out[:, -1, :])

        return out.view(-1)

'''
class CRNN(nn.Module):

    def __init__(self, variables, channels, vector_dim, rnn_hidden_size, rnn_num_layers, dropout=None):

        super(CRNN, self).__init__()

        # CNN
        self.cnn = CNN(variables, channels, vector_dim, dropout)
        # RNN
        self.rnn = RNN(vector_dim, rnn_hidden_size, rnn_num_layers)

    def forward(self, input):
        # CNN
        batch_size, shape1, T, H, W = input.shape
        input = input.reshape(batch_size*T, shape1, H, W)
        x = self.cnn(input)
        x = x.reshape(batch_size, T, x.shape[-1])
        # RNN
        out = self.rnn(x)
        return out.view(-1)

'''
'''
For CRNN many-to-many we just need to change the output of self.fc2 to be 6 instead of 1,
 and return out instead of out.view(-1)
'''


class CNN_Unet(nn.Module):

    def __init__(self, variables, channels, vector_dim, dropout):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CNN_Unet, self).__init__()

        # CNN
        self.conv1 = nn.Conv2d(variables, 32, 3, stride=1, padding=1)    # output: (batch,96,192,64)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)  #  (batch,96,192,64)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)      #  (batch,48,96,64)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)    # output: (batch,48,96,128)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  #  (batch,48,96,128)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)         #  (batch,24,48,128)

        self.conv5 = nn.Conv2d(64, 96, 3, stride=1, padding=1)    # output: (batch,24,48,256)
        self.conv6 = nn.Conv2d(96, 96, 3, stride=1, padding=1)  #  (batch,24,48,256)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)      #  (batch,12,24,256)

        self.conv7 = nn.Conv2d(96, 128, 3, stride=1, padding=1)    # output: (batch,12,24,512)
        self.conv8 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  #  (batch,12,24,512)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)         #  (batch,6, 12, 512)

        # Linear
        self.fc1 = nn.Linear(128*6*12, 500)


    def forward(self, input):

        # CNN
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.max1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.max3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.max4(x)
        # resize tensor and FC
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)

        return x

class ResNetCRNN(nn.Module):

    def __init__(self, variables, channels, vector_dim, rnn_hidden_size, rnn_num_layers, dropout=None):

        super(ResNetCRNN, self).__init__()

        # CNN
        # ResNet uses inputs size 224x224
        self.resnet = models.resnet50()
        # RNN
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(1280, 500)
        #self.conv = nn.Sequential(*list(self.resnet.children())[:-1])



        self.rnn = RNN(vector_dim, rnn_hidden_size, rnn_num_layers)

    def forward(self, input):
        # CNN
        batch_size, shape1, T, H, W = input.shape
        input = input.reshape(batch_size*T, shape1, H, W)
        x = self.resnet(input)
        x = x.reshape(batch_size, T, x.shape[-1])
        # RNN
        out = self.rnn(x)
        return out.view(-1)


'''
Here we can add the metrics we decide to use.
'''
