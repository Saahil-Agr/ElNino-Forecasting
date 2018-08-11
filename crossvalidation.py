import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from train import train
import model.crnn as net
import torch
import torch.optim as optim
import logging
import utils
import torch.nn as nn
from evaluate import evaluate


# load the 70 years of data and its 3 months ahead labels.
def load_data(data_dir, labels_dir):
    # load inputs
    images_path = data_dir
    filenames = sorted(os.listdir(images_path))
    filenames = [os.path.join(images_path, f) for f in filenames if f.endswith('.npy')]
    inputs = [np.load(filenames[idx]) for idx in range(len(filenames))]
    inputs = [np.expand_dims(input, axis = 0) for input in inputs]
    inputs = [torch.FloatTensor(input) for input in inputs]
    # load labels
    labels_path = os.path.join(data_dir, labels_dir)
    labels_df = pd.read_csv(labels_path)
    labels = np.asarray(labels_df['skt_detrend'].tolist())
    return inputs, labels

# dataloader for each cross-validation train and evaluation
class CrossDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        # return size of dataset
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]  # return both as tensors

# restore parameters from file
def load_parameters(old_checkpoint_dir, restore_file, model):
    restore_path = os.path.join(old_checkpoint_dir, restore_file + '.pth.tar')
    logging.info("Restoring parameters from {}".format(restore_path))
    utils.load_checkpoint(restore_path, model)
    logging.info("Parameters loaded")

def create_dataloaders(all_inputs, all_labels, current_pos):
    eval_inputs = all_inputs[current_pos*12:(current_pos+1)*12]
    eval_labels = all_labels[current_pos*12:(current_pos+1)*12]
    train_inputs = all_inputs[:]
    train_labels = list(all_labels[:])

    print('New validation inputs, from ' + str(current_pos*12) + ' to ' + str((current_pos+1)*12-1))

    if current_pos == 0:
        del train_inputs[:24]    # remove first 2 years
        del train_labels[:24]
        print('deleted first 24 values')
    elif current_pos == len(all_inputs)/12 - 1:
        del train_inputs[-24:]        #remove last 2 years
        del train_labels[-24:]
        print('deleted last 24 values')
    else:
        del train_inputs[(current_pos-1)*12:(current_pos+2)*12]    # remove evaluation year and 2 years next to it
        del train_labels[(current_pos-1)*12:(current_pos+2)*12]
        print('deleted values from ' + str((current_pos-1)*12) + ' to ' + str((current_pos+2)*12-1))
    # create train and val dataloaders:
    print('Number of training inputs: ' + str(len(train_labels)))
    train_dl = DataLoader(CrossDataset(train_inputs, train_labels), batch_size=64, shuffle=True)
    val_dl = DataLoader(CrossDataset(eval_inputs, eval_labels), batch_size=64, shuffle=False)
    return train_dl, val_dl

# save cross-validation history in csv file:
def save_crossval(current_pos, train_MSE, val_MSE, train_L1, val_L1, results_dir):
    crossval_path = os.path.join(results_dir,'crossval_results.csv')
    # if history file doesn't exist yet, create the dataframe with the header:
    print('current position: '+str(current_pos))
    if current_pos == 0:
        crossval_df = pd.DataFrame(columns=['year','trainMSE','valMSE','trainL1','valL1','trainRMSE','valRMSE'])
    else:
        crossval_df = pd.read_csv(crossval_path)
    crossval_df.loc[current_pos] = [current_pos+1, train_MSE, val_MSE, train_L1, val_L1, np.sqrt(train_MSE), np.sqrt(val_MSE)]
    crossval_df.to_csv(crossval_path, index=False)





def main():

    data_dir = 'data/3d_real/all_6months'
    old_checkpoint_dir = 'experiments/6month_GCM_training/checkpoint_6m_drop0.1'
    results_dir = 'crossval_results_6m'
    labels_dir = 'labels6m.csv'

    all_inputs, all_labels = load_data(data_dir, labels_dir)
    print('Number of inputs: ' + str(len(all_inputs)))
    print('Number of labels: ' + str(len(all_labels)))

    # check if the inputs and labels are multiples of 12 (complete years)
    if len(all_inputs)%12 != 0 or len(all_labels)%12 != 0:
        print('data must be complete years')
        return

    # define model parameters:
    channels = 10
    vector_dim = 500
    rnn_hidden_size = 500
    rnn_num_layers = 2
    dropout = 0.1
    lrate = 0.00001
    lr_step_size = 10
    lr_factor = 0.2
    epochs = 50
    restore_file = 'best'
    dtype = torch.float32

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Im using GPU')
    else:
        device = torch.device('cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # create model and loss instance
    model = net.CRNN(channels, vector_dim, rnn_hidden_size, rnn_num_layers, dropout=dropout)
    loss_fn = net.loss_fn

    # Set the logger, will be saved in the results folder
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    utils.set_logger(os.path.join(results_dir, 'train.log'))

    logging.info("learning rate: " + str(lrate))
    logging.info("learning rate step size: " + str(lr_step_size))
    logging.info("learning rate factor: " + str(lr_factor))
    logging.info("epochs: " + str(epochs))

    '''
    Start cross-validation
    '''

    counter = 0

    # iterate for year after year
    for i in range(int(len(all_inputs)/12)):

        # create current dataloaders
        train_dl, val_dl = create_dataloaders(all_inputs, all_labels, i)

        # initialize model and optimizer for every new training set
        model.apply(net.initialize_weights)
        params = model.parameters()
        optimizer = optim.Adam(params, lr=lrate, betas=(0.9, 0.999))
        optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_factor, last_epoch=-1)

        # reload weights from restore_file if specified
        if restore_file is not None:
            load_parameters(old_checkpoint_dir, restore_file, model)



        logging.info('Start training for year: ' + str(i+1))

        for epoch in range(epochs):

            if epoch == 0:
                train_MSE, counter = evaluate(counter, 'train', i, epoch, results_dir, model, nn.MSELoss(), train_dl, device, dtype)
                val_MSE, counter = evaluate(counter, 'val', i, epoch, results_dir, model, nn.MSELoss(), val_dl, device, dtype)
                logging.info("- Initial Train average RMSE loss: " + str(np.sqrt(train_MSE)))
                logging.info("- Initial Validation average RMSE loss: " + str(np.sqrt(val_MSE)))

            print("Epoch {}/{}".format(epoch + 1, epochs))
            train(model, optimizer, loss_fn, train_dl, device, dtype)

            if i < 1:
                logging.info("Epoch {}/{}".format(epoch + 1, epochs))
                # Evaluate MSE for one epoch on train and validation set for the first training
                train_MSE, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.MSELoss(), train_dl, device, dtype)
                val_MSE, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.MSELoss(), val_dl, device, dtype)
                train_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), train_dl, device, dtype)
                val_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), val_dl, device, dtype)
                logging.info("- Train average RMSE loss: " + str(np.sqrt(train_MSE)))
                logging.info("- Validation average RMSE loss: " + str(np.sqrt(val_MSE)))
                logging.info("- Train average L1 loss: " + str(train_L1))
                logging.info("- Validation average L1 loss: " + str(val_L1))


            if epoch%10 == 0:
                logging.info("Epoch {}/{}".format(epoch + 1, epochs))
                train_MSE, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.MSELoss(), train_dl, device, dtype)
                val_MSE, counter = evaluate(counter, 'val', i, epoch+1, results_dir, model, nn.MSELoss(), val_dl, device, dtype)
                train_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), train_dl, device, dtype)
                val_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), val_dl, device, dtype)
                logging.info("- Train average RMSE loss: " + str(np.sqrt(train_MSE)))
                logging.info("- Validation average RMSE loss: " + str(np.sqrt(val_MSE)))
                logging.info("- Train average L1 loss: " + str(train_L1))
                logging.info("- Validation average L1 loss: " + str(val_L1))


        # Evaluate and save MSE and L1  at the end of each training
        train_MSE, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.MSELoss(), train_dl, device, dtype)
        val_MSE, counter = evaluate(counter, 'val', i, epoch+1, results_dir, model, nn.MSELoss(), val_dl, device, dtype)
        train_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), train_dl, device, dtype)
        val_L1, counter = evaluate(counter, 'train', i, epoch+1, results_dir, model, nn.L1Loss(), val_dl, device, dtype)

        save_crossval(i, train_MSE, val_MSE, train_L1, val_L1, results_dir)


if __name__ == '__main__':

    main()










