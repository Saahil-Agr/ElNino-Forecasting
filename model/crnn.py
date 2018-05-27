
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

    def __init__(self, channels, vector_dim):

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
        self.fc = nn.Linear(channels*16*3*9, vector_dim)

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

    def __init__(self, channels, vector_dim):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN, self).__init__()

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

        self.fc = nn.Linear(channels*16*3*9, vector_dim)
        #self.fc1_drop = nn.Dropout(p=0.5)


        #self.lstm1 = nn.LSTM(input_size=1, hidden_size=64) # input_dim = 1, output_dim = 64
        #self.lstm2 = nn.LSTM(input_size=64,hidden_size=1) # input_dim = 64, output_dim = 1

    def forward(self, input):


        print(input.shape)
        x = F.relu(self.conv1_bn(self.conv1(input)))
        #print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print(x.shape)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print(x.shape)
        x = F.relu(self.conv6(x))
        #print(x.shape)

        # resize tensor to fit the FC
        x = x.view(x.size()[0], -1)
        #print(x.shape)
        x = self.fc(x)
        #print(x.shape)
        #x = F.relu(self.fc(x))

        # lstm inputs: 3D, 1st sequence (time_steps), 2nd minibatch instances, 3rd elements of the inputs (vectors of each time_step).
        #lstm1_out, hidden_state1 = self.lstm1(input, (h0, c0))
        #lstm2_out, hidden_state2 = self.lstm2(input, (h0, c0))

        return x.view(-1) #lstm1_out

'''
Here we can add the metrics we decide to use.
'''
