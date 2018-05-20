
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

'''
Convolutional recurrent neural net class, loss and metrics.
'''

class CRNN(nn.Module):

    def __init__(self, channels, vector_dim):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, 3, 1, 1)    # output: (batch,64,64,32)

        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, 4, 2, 1)  #  (batch,32,32,64)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.conv2_bn = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels*4, 4, 2, 1) # (batch,16,16,128)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.conv3_drop = nn.Dropout2d(p=0.5)
        self.conv3_bn = nn.BatchNorm2d(channels*4)

        self.conv4 = nn.Conv2d(channels*4, channels*8, 4, 2, 1) # (batch,8,8,256)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.conv4_drop = nn.Dropout2d(p=0.5)
        self.conv4_bn = nn.BatchNorm2d(channels*8)

        self.conv5 = nn.Conv2d(channels*8, channels*16, 4, 2, 1) # (batch,4,4,512)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(channels*16)

        self.conv6 = nn.Conv2d(channels*16, channels*32, 4, 1, 0) # (batch,1,1,1024)
        nn.init.kaiming_normal_(self.conv6.weight)
        self.conv6_drop = nn.Dropout2d(p=0.5)

        self.fc = nn.Linear(channels*32, vector_dim)
        nn.init.kaiming_normal_(self.fc.weight)
        self.fc1_drop = nn.Dropout(p=0.5)

        # weight initialization
        #for m in self.modules():
         #   nn.init.kaiming_normal_(m.weight)


        #self.lstm1 = nn.LSTM(input_size=1, hidden_size=64) # input_dim = 1, output_dim = 64
        #self.lstm2 = nn.LSTM(input_size=64,hidden_size=1) # input_dim = 64, output_dim = 1

    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(input)))
        x = F.relu(self.conv2_bn(self.conv2_drop(self.conv2(x))))
        x = F.relu(self.conv3_bn(self.conv3_drop(self.conv3(x))))
        x = F.relu(self.conv4_bn(self.conv4_drop(self.conv4(x))))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_drop(self.conv6(x)))

        # resize tensor to fit the FC
        x = x.view(x.size()[0], -1)
        x = self.fc1_drop(self.fc(x))
        #x = F.relu(self.fc(x))

        # lstm inputs: 3D, 1st sequence (time_steps), 2nd minibatch instances, 3rd elements of the inputs (vectors of each time_step).
        #lstm1_out, hidden_state1 = self.lstm1(input, (h0, c0))
        #lstm2_out, hidden_state2 = self.lstm2(input, (h0, c0))

        return x.view(-1) #lstm1_out


def loss_fn(prediction, target):

    loss = nn.MSELoss()
    output = loss(prediction, target)

    return output


'''
Convolutional recurrent neural net class, loss and metrics.
'''

class CRNN2(nn.Module):

    def __init__(self, channels, vector_dim):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(CRNN2, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, 3, stride=1, padding=1)    # output: (batch,96,192,32)

        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv1_bn = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, 4, 2, 1)  #  (batch,48,96,64)
        nn.init.kaiming_normal_(self.conv2.weight)
        #self.conv2_drop = nn.Dropout2d(p=0.5)
        self.conv2_bn = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels*4, 4, 2, 1) # (batch,24,48,128)
        nn.init.kaiming_normal_(self.conv3.weight)
        #self.conv3_drop = nn.Dropout2d(p=0.5)
        self.conv3_bn = nn.BatchNorm2d(channels*4)

        self.conv4 = nn.Conv2d(channels*4, channels*8, 4, 2, 1) # (batch,12,24,256)
        nn.init.kaiming_normal_(self.conv4.weight)
        #self.conv4_drop = nn.Dropout2d(p=0.5)
        self.conv4_bn = nn.BatchNorm2d(channels*8)

        self.conv5 = nn.Conv2d(channels*8, channels*16, 4, 2, 1) # (batch,6,12,512)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm2d(channels*16)

        self.conv6 = nn.Conv2d(channels*16, channels*16, 4, 1, 0) # (batch,3,6,1024)
        nn.init.kaiming_normal_(self.conv6.weight)
        #self.conv6_drop = nn.Dropout2d(p=0.5)

        self.fc = nn.Linear(channels*16*3*9, vector_dim)
        nn.init.kaiming_normal_(self.fc.weight)
        #self.fc1_drop = nn.Dropout(p=0.5)

        # weight initialization
        #for m in self.modules():
         #   nn.init.kaiming_normal_(m.weight)


        #self.lstm1 = nn.LSTM(input_size=1, hidden_size=64) # input_dim = 1, output_dim = 64
        #self.lstm2 = nn.LSTM(input_size=64,hidden_size=1) # input_dim = 64, output_dim = 1

    def forward(self, input):
        #print(input.shape)
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
