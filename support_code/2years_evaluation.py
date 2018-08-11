import os
import pandas as pd
import numpy as np
import torch
import model.crnn as net
import utils
import torch.optim as optim
import matplotlib.pyplot as plt

def create_dataset(months):
    # load files
    labels_path = os.path.join(data_dir, "{}".format('labels.csv'))
    filenames = sorted(os.listdir(data_dir))
    filenames = [os.path.join(data_dir, f) for f in filenames if f.endswith('.npy')]
    #create labels
    labels_df = pd.read_csv(labels_path)
    labels = np.asarray(labels_df['tas'].tolist())[0:months]
    labels = torch.FloatTensor(labels)
    #create input images
    inputs = [np.load(filenames[idx]) for idx in range(len(filenames))]
    inputs = [np.expand_dims(input, axis = 0) for input in inputs]
    inputs = np.asarray(inputs)[0:months]
    inputs = torch.FloatTensor(inputs)
    return inputs, labels

def load_model(checkpoint_dir, restore_file):
    restore_path = os.path.join(checkpoint_dir, restore_file + '.pth.tar')
    utils.load_checkpoint(restore_path, model, optimizer)

def evaluate(inputs, labels, model):

    model.eval()
    with torch.no_grad():
        predictions = model(inputs)
    return predictions

def plot_predictions(predictions1, labels, predictions2):
    plt.plot(predictions1, color='green')
    plt.plot(predictions2, color='red')
    plt.plot(labels, color='black')
    plt.legend(['predictions CRNN','predictions 3D CNN', 'real values'])
    plt.xlabel('Months')
    plt.ylabel('Temperature Anomalies')
    plt.show()

if __name__ == '__main__':

    data_dir = 'data/3d/test/full_data'
    lr = 0.00001
    channels = 10
    vector_dim = 500
    rnn_hidden_size = 500
    rnn_num_layers = 2
    device = torch.device('cpu')

    #checkpoint_dir = 'gcloud/checkpoint_500vector'
    checkpoint_dir = 'gcloud/checkpoint_3months'
    restore_file = 'best'
    model = net.CRNN(channels, vector_dim, rnn_hidden_size, rnn_num_layers)
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    predictions2 = np.load('3DCNN/3m_test_predictions.npy')
    #predictions2 = np.load('3DCNN/1m_test_predictions.npy')



    inputs, labels = create_dataset(12*10)
    load_model(checkpoint_dir, restore_file)
    predictions = evaluate(inputs, labels, model)
    plot_predictions(predictions.numpy(), labels.numpy(), predictions2)

    #plt.savefig('10 years validation 1m ahead')
