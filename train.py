
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.crnn as net
from model import data_loader
from evaluate import evaluate
import matplotlib.pyplot as plt
import torch.nn as nn

import pandas as pd


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
               # print('Iteration %d, loss = %.4f' % (i, loss.item()))



            # update the average loss
            #loss_avg = torch.mean(torch.FloatTensor(losses))
            #logging.info("- Train average loss : " + loss_avg)
            # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            t.update()

    return np.mean(losses), losses


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs,
                       restore_file=None):

    epoch_train_losses = []
    batch_train_losses = []
    val_losses = []

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(checkpoint_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = float('inf')


    for epoch in range(epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        loss_avg_epoch, loss_avg_batch = train(model, optimizer, loss_fn, train_dataloader)
        epoch_train_losses.append(loss_avg_epoch)
        batch_train_losses += loss_avg_batch

        logging.info("- Training average loss : " + str(loss_avg_epoch))

        # Evaluate for one epoch on validation set
        val_loss_avg = evaluate(model, loss_fn, val_dataloader, device, dtype)
        val_losses.append(val_loss_avg)

        logging.info("- Validation average loss : " + str(val_loss_avg))

        is_best = val_loss_avg <= best_val_loss

        model.eval()
        error = 0
        num_samples = 0
        with torch.no_grad():
            for x, y in train_dataloader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)
                scores = model(x)
                error += np.sum(np.square(scores.numpy() - y.numpy()))
                num_samples += scores.shape[0]
            total_error = float(error) / num_samples
            print('Got mean train error of ' + str(total_error))

        model.eval()
        error = 0
        num_samples = 0
        with torch.no_grad():
            for x, y in val_dataloader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)
                scores = model(x)
                error += np.sum(np.square(scores.numpy() - y.numpy()))
                num_samples += scores.shape[0]
            total_error = float(error) / num_samples
            print('Got mean val error of ' + str(total_error))




        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best evaluation loss")
            best_val_ = val_loss_avg

            # Save best val loss in a json file in the checkpoint directory
            best_val_path = "best_val_loss.txt"
            utils.save_dict_to_txt(val_loss_avg, results_dir, best_val_path, epoch)

        # Save latest val metrics in a json file in the checkpoint directory
        last_val_path = "last_val_loss.txt"
        utils.save_dict_to_txt(val_loss_avg, results_dir, last_val_path, epoch)


        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=checkpoint_dir)


    # plot training history
    utils.show_train_hist(epoch_train_losses, results_dir, show=True, epoch_plot=True, save=True)
    utils.show_train_hist(batch_train_losses, results_dir, show=True, epoch_plot=False, save=True)
    utils.show_train_val_hist(epoch_train_losses, val_losses, results_dir, show=True, save=True)



if __name__ == '__main__':


    import os
    from PIL import Image

    '''
    filenames = os.listdir('data/train/img')
    filenames = [os.path.join('data/train/img', f) for f in filenames if f.endswith('.npy')]
    input = np.load(filenames[10])
    print(type(input))
    print(input.shape)
    print(np.min(input))
    print(np.max(input))
    print(np.mean(input))
    print(input)
    '''

    # training hyperparameters
    batch_size = 64
    lr = 0.000008
    epochs = 30
    model_dir = 'model'
    model_name = 'crnn'
    checkpoint_dir = 'checkpoint'
    results_dir = 'results'
    print('learning rate: ', lr)
    print('epochs: ', epochs)
    #model parameters
    channels = 10
    vector_dim = 1
    restore_file=None

    USE_GPU = True
    dtype = torch.float32
    print_every = 1    # iterations before printing

    # use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Define the model, dataset and optimizer
    if model_name == 'cnn':
        model = net.CNN(channels, vector_dim)
        data_dir = 'data/2d'
        logging.info("model: CNN")
    if model_name == 'crnn':
        model = net.CRNN(channels, vector_dim)
        data_dir = 'data/3d'
        logging.info("model: CRNN")
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


