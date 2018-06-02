
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.ThreeDcnn as ThreeDcnn
from model import data_loader_3d
from evaluate import evaluate

#import matplotlib


import pandas as pd



def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, e, best):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    losses = []
    eval_every = 12
    n = len(train_dataloader)
    # Use tqdm for progress bar
    with tqdm(total=len(train_dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_dataloader):

            model.train()  # put model to training mode

            train_batch = train_batch.to(device=device, dtype=dtype)
            labels_batch = labels_batch.to(device=device, dtype=dtype)

            # compute forward pass
            #print("train batch",train_batch.shape)
            output_batch = model(train_batch)
            #print("output batch",output_batch.shape,'\n',labels_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            # save each batch loss, this can be done also once in a while
            losses.append(loss.item())
            # performs updates using calculated gradients
            optimizer.step()

            if i % print_every == 0:
                #print('Iteration %d, loss = %.4f' % (i+e*n, loss.item()))
                with open(os.path.join(results_dir,"train_loss.txt"), "a") as f:
                    f.write("{}, {} \n".format(i+e*n,loss.item()))


            if i % eval_every == 0 and i != 0:
                logging.info("- Iteration %d, Evaluating on validation set.." % (i+e*n))
                val_loss_avg = evaluate(model, loss_fn, val_dataloader, device, dtype)

                with open(os.path.join(results_dir,"val_loss.txt"), "a") as f:
                    f.write("Iteration {}, loss {} \n".format(i+e*n,val_loss_avg))

                logging.info("- Intermediate Validation loss : " + str(val_loss_avg))
                is_best = val_loss_avg <= best
                if is_best:
                    logging.info("- Found new best accuracy")
                    best = val_loss_avg

                    # Save best val loss in a text file in the checkpoint directory
                    with open(os.path.join(results_dir, "best_val_loss.txt"), "a") as f:
                        f.write("Iteration {}, best loss {} \n".format(i+e*n, val_loss_avg))

                    utils.save_checkpoint({'iter': i, 'epoch': e,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict(),
                                           'best_loss': val_loss_avg},
                                          is_best=is_best,
                                          checkpoint=checkpoint_dir)
            t.update()

    return losses, best


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs,
                       restore_file=None):

    total_batch_loss = []
    val_losses = []
    start_epoch = 0
    best_val_loss = float('inf')
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(checkpoint_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = utils.load_checkpoint(restore_path, model, optimizer)
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_loss',float('inf'))



    for epoch in range(epochs + start_epoch):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        batch_loss, best_temp = train(model, optimizer, loss_fn, train_dataloader, val_dataloader, epoch, best_val_loss)
        best_val_loss = best_temp
        total_batch_loss += batch_loss

        # Evaluate for one epoch on validation set
        logging.info("- Training average loss : " + str(val_MSE))

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
        val_losses.append(val_MSE)

        logging.info("- Validation average loss : " + str(val_MSE))

        is_best = val_MSE <= best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=checkpoint_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_loss = val_MSE

            # Save best val loss in a text file in the checkpoint directory
            best_val_path = "val_loss.txt"
            utils.save_dict_to_txt(val_MSE, results_dir, best_val_path, epoch)
            utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'best_loss' : val_MSE},
                               is_best=is_best,
                               checkpoint=checkpoint_dir)


        ## plots of losses
        if epoch !=0 or restore_file is not None:
            epoch_train_losses = np.load(os.path.join(results_dir, "epoch_avg_trainloss.npy"))

        np.save(os.path.join(results_dir,"epoch_avg_trainloss"), epoch_train_losses)
        np.save(os.path.join(results_dir, "epoch_val_loss"), val_losses)
    utils.show_train_hist(total_batch_loss, results_dir, show=False, epoch_plot=False, save=True)
    #utils.show_train_val_hist(epoch_train_losses, val_losses, results_dir, show=False, save=True)

if __name__ == '__main__':

    '''
    Main file for running the training. Initializes all the required variables and flags
    '''
    import os
    from PIL import Image

    # training hyperparameters
    batch_size = 32
    lr = 0.0002
    epochs = 8
    data_dir = 'data/3d'
    model_dir = 'model'
    checkpoint_dir = '3d_checkpoint/small_exp1'
    results_dir = '3d_results/small_exp1'
    print('learning rate', lr)
    print('epochs', epochs)
    #model parameters
    channels = 10
    vector_dim = 1
    restore = None #'last'
    USE_GPU = True
    dtype = torch.float32
    print_every = 1    # iterations before printing
    #main_dir = os.path.relpath(os.path.dirname(__file__))
    # use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # Set the logger
    if not os.path.exists(results_dir):
        print("Results Directory does not exist! Making directory {}".format(results_dir))
        os.mkdir(results_dir)
    utils.set_logger(os.path.join(results_dir, 'train.log'))

    # Define the model and optimizer
    Dcnn_model = ThreeDcnn.ThreeDCNN(channels, vector_dim)
    optimizer = optim.Adam(Dcnn_model.parameters(), lr=lr, betas=(0.5, 0.999))
    Dcnn_model = Dcnn_model.to(device=device)  # move the model parameters to CPU/GPU
    # fetch loss function and metrics
    loss_fn = ThreeDcnn.loss_fn

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader_3d.fetch_dataloader(['train','val'], data_dir, batch_size)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    #test_dl = dataloaders['test']

    logging.info("- done.")

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(epochs))
    train_and_evaluate(Dcnn_model, train_dl, val_dl, optimizer, loss_fn, epochs, restore_file = restore)
