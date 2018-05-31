import logging
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import utils
import model.data_loader as data_loader
import model.crnn as net
from evaluate import evaluate
import torch.nn as nn
import os


def train(model, optimizer, loss_fn, dataloader):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    losses = []

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            model = model.to(device=device)  # move the model parameters to CPU/GPU
            model.train()  # put model to training mode

            train_batch = train_batch.to(device=device, dtype=dtype)
            labels_batch = labels_batch.to(device=device, dtype=dtype)

            # compute forward pass
            output_batch = model(train_batch)

            #criterion = nn.L1Loss()
            #loss = criterion(output_batch, labels_batch)
            loss = loss_fn(output_batch, labels_batch)

            # save each batch loss, this can be done also once in a while
            losses.append(loss.item())

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # in case you want to print the training loss after every n iterations
            #if i % print_every == 0:
             #   print('Iteration %d, loss = %.4f' % (i, loss.item()))

            t.update()

    return np.mean(losses), losses


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs,
                       restore_file=None):

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(checkpoint_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_MSE = float('inf')

    for epoch in range(epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader)

        # Evaluate MSE for one epoch on train and validation set
        train_MSE = evaluate(model, nn.MSELoss(), train_dataloader, device, dtype)
        val_MSE = evaluate(model, nn.MSELoss(), val_dataloader, device, dtype)
        # Evaluate L1 for one epoch on train and validation set
        train_L1 = evaluate(model, nn.L1Loss(), train_dataloader, device, dtype)
        val_L1 = evaluate(model, nn.L1Loss(), val_dataloader, device, dtype)

        # save training history in csv file:
        utils.save_history(epoch, train_MSE, val_MSE, train_L1, val_L1, results_dir)

        # print losses
        logging.info("- Train average MSE loss: " + str(train_MSE))
        logging.info("- Validation average MSE loss: " + str(val_MSE))
        logging.info("- Train average L1 error: " + str(train_L1))
        logging.info("- Validation average L1 error: " + str(val_L1))

        # save MSE if is the best
        is_best = val_MSE <= best_val_MSE
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best evaluation loss")
            # Save best val loss in a txt file in the checkpoint directory
            best_val_path = "best_val_loss.txt"
            utils.save_dict_to_txt(val_MSE, results_dir, best_val_path, epoch)

        # Save latest val metrics in a json file in the checkpoint directory
        last_val_path = "last_val_loss.txt"
        utils.save_dict_to_txt(val_MSE, results_dir, last_val_path, epoch)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=checkpoint_dir)

if __name__ == '__main__':

    #directories
    model_dir = 'model'
    checkpoint_dir = 'checkpoint'
    results_dir = 'results'

    # choose model
    model_name = 'cnn'

    # training hyperparameters
    batch_size = 64
    lr = 0.00001
    epochs = 3
    channels = 10

    # hyperparameters for CRNN
    vector_dim = 20
    rnn_hidden_size = 500
    rnn_num_layers = 2

    # other parameters
    restore_file=None
    USE_GPU = True
    dtype = torch.float32
    print_every = 1    # iterations before printing


    # use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print('Im using GPU')
    else:
        device = torch.device('cpu')

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # Set the logger, will be saved in the results folder
    utils.set_logger(os.path.join(results_dir, 'train.log'))

    # print hyperparameters
    logging.info("learning rate: " + str(lr))
    logging.info("epochs: " + str(epochs))
    logging.info("batch size: " + str(batch_size))
    logging.info("first layer filters: " + str(channels))

    # Define the model, dataset and optimizer
    if model_name == 'cnn':
        model = net.CNN(channels)
        data_dir = 'data/2d'
        logging.info("model: CNN")
    if model_name == 'crnn':
        model = net.CRNN(channels, vector_dim, rnn_hidden_size, rnn_num_layers)
        data_dir = 'data/3d'
        logging.info("model: CRNN")
        logging.info("encoding dimesion: " + str(vector_dim))
        logging.info("RNN layers: " + str(rnn_num_layers))
        logging.info("RNN hidden units: " + str(rnn_hidden_size))

    model.apply(net.initialize_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # fetch loss function
    loss_fn = net.loss_fn

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], data_dir, batch_size, model_name)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    #test_dl = dataloaders['test']

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, epochs, restore_file=restore_file)


