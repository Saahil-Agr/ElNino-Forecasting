
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init

'''
Convolutional recurrent neural net class, loss and metrics.
'''

class ThreeDCNN(nn.Module):

    def __init__(self, channels, vector_dim):

        # convolution layers arguments: input_channels, output_channels, filter_size, stride, padding.
        super(ThreeDCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, channels, (3,3,2), (1,1,2), (1,1,0))    # output: (batch,c,24,96,96)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv3d(channels, channels*2, (3,4,4), (1,2,2), 1)  #  (batch,2c,24,48,48)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.conv2_bn = nn.BatchNorm3d(channels*2)
        self.conv3 = nn.Conv3d(channels*2, channels*4, (3,4,4), (1,2,2), 1) # (batch,4c, 24,24,24)
        nn.init.kaiming_normal_(self.conv3.weight)
        self.conv3_bn = nn.BatchNorm3d(channels*4)
        self.conv4 = nn.Conv3d(channels*4, channels*8, 4, 2, 1) # (batch,8c,12,12,12)
        nn.init.kaiming_normal_(self.conv4.weight)
        self.conv4_bn = nn.BatchNorm3d(channels*8)
        self.conv5 = nn.Conv3d(channels*8, channels*16, 6, 2, 0) # (batch,16c,4,4,4)
        nn.init.kaiming_normal_(self.conv5.weight)
        self.conv5_bn = nn.BatchNorm3d(channels*16)
        self.conv6 = nn.Conv3d(channels*16, channels*32, 4, 1, 0) # (batch,1024,1,1,1)
        nn.init.kaiming_normal_(self.conv6.weight)

        self.fc = nn.Linear(channels*32, vector_dim)
        nn.init.kaiming_normal_(self.fc.weight)

        # weight initialization
        #for m in self.modules():
         #   nn.init.kaiming_normal_(m.weight)


        #self.lstm1 = nn.LSTM(input_size=1, hidden_size=64) # input_dim = 1, output_dim = 64
        #self.lstm2 = nn.LSTM(input_size=64,hidden_size=1) # input_dim = 64, output_dim = 1

    def forward(self, input):
        #print(input.shape)
        x = F.relu(self.conv1(input))
        #print("conv1",x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        #print("conv2", x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print("conv3", x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        #print("conv4", x.shape)
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #print("conv5", x.shape)
        x = F.relu(self.conv6(x))
        #print("conv6", x.shape)

        # resize tensor to fit the FC
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        # lstm inputs: 3D, 1st sequence (time_steps), 2nd minibatch instances, 3rd elements of the inputs (vectors of each time_step).
        #lstm1_out, hidden_state1 = self.lstm1(input, (h0, c0))
        #lstm2_out, hidden_state2 = self.lstm2(input, (h0, c0))

        #print("x.view",x.view(-1))
        return x.view(-1) #lstm1_out


def loss_fn(prediction, target):

    loss = nn.MSELoss()
    output = loss(prediction, target)

    return output


'''
Here we can add the metrics we decide to use.
'''
