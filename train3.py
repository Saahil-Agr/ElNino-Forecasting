
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.ThreeDcnn as ThreeDcnn
from model import data_loader_3d
from evaluate import evaluate

import pandas as pd



def train(model, optimizer, loss_fn, train_dataloader, val_dataloader,e):
    # set model to training mode
    model.train()
    e += 1
    # summary for current training loop and a running average object for loss
    losses = []
    val_loss = []
    eval_every = 50
    best_val_loss = float('inf')
    # Use tqdm for progress bar
    with tqdm(total=len(train_dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_dataloader):

            model = model.to(device=device)  # move the model parameters to CPU/GPU
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

            # performs updates using calculated gradients
            optimizer.step()

            if i % print_every == 0:
                print('Iteration %d, loss = %.4f \n' % (i+1, loss.item()))
                with open("train_loss.txt", "a") as f:
                    f.write("{}, {} \n".format(i+1,loss.item()))

            # save each batch loss, this can be done also once in a while
            losses.append(loss.item())
            if i % eval_every == 0:
                logging.info("- Iteration %d, Evaluating on validation set.. \n" % i)
                val_loss_avg = evaluate(model, loss_fn, val_dataloader, device, dtype)
                val_loss.append(val_loss_avg)

                with open("val_loss.txt", "a") as f:
                    f.write("Iteration {}, loss {} \n".format(i+1,val_loss_avg))

                logging.info("- Inermediate Validation loss : " + str(val_loss_avg))
                is_best = val_loss_avg <= best_val_loss
                if is_best:
                    logging.info("- Found new best accuracy")
                    best_val_loss = val_loss_avg

                    # Save best val loss in a text file in the checkpoint directory
                    best_val_path = "best_val_loss.txt"
                    utils.save_dict_to_txt(val_loss_avg, results_dir, best_val_path, iteration = i+1)
                    utils.save_checkpoint({'iter': i,
                                           'state_dict': model.state_dict(),
                                           'optim_dict': optimizer.state_dict()},
                                          is_best=is_best,
                                          checkpoint=checkpoint_dir)

        # update the average loss
        #loss_avg = torch.mean(torch.FloatTensor(losses))
        #logging.info("- Train average loss : " + loss_avg)
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    return np.mean(losses), losses, val_loss


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs,
                       restore_file=None):

    epoch_train_losses = []
    batch_train_losses = []
    val_losses = []
    val_losses_batch = []
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
        loss_avg_epoch, loss_avg_batch, batch_val_loss = train(model, optimizer, loss_fn, train_dataloader, val_dataloader,epoch)
        epoch_train_losses.append(loss_avg_epoch)
        batch_train_losses += loss_avg_batch
        val_losses_batch += batch_val_loss

        # Evaluate for one epoch on validation set
        logging.info("- Training average loss : " + str(loss_avg_epoch))

        # Evaluate for one epoch on validation set
        val_loss_avg = evaluate(model, loss_fn, val_dataloader, device, dtype)
        val_losses.append(val_loss_avg)

        logging.info("- Validation average loss : " + str(val_loss_avg))

        is_best = val_loss_avg <= best_val_loss

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=checkpoint_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_loss = val_loss_avg

            # Save best val loss in a text file in the checkpoint directory
            best_val_path = "best_val_loss.txt"
            utils.save_dict_to_txt(val_loss_avg, results_dir, best_val_path, epoch)

            # Save best val metrics in a json file in the model directory
            #best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            #utils.save_dict_to_json(val_loss_avg, best_json_path)

        # Save latest val metrics in a text file in the checkpoint directory
        last_val_path = "last_val_loss.txt"
        utils.save_dict_to_txt(val_loss_avg, results_dir, last_val_path, epoch)

        model.eval()
        error = 0
        num_samples = 0
        with torch.no_grad():
            for x, y in train_dataloader:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=device, dtype=dtype)
                scores = model(x)
                error += np.sum(np.square(scores.cpu().numpy() - y.cpu().numpy()))
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
                error += np.sum(np.square(scores.cpu().numpy() - y.cpu().numpy()))
                num_samples += scores.shape[0]
            total_error = float(error) / num_samples
            print('Got mean val error of ' + str(total_error))

        ## plots of losses
       # utils.show_train_hist(batch_train_losses, results_dir, show=False, epoch_plot=False, save=True)
       # utils.show_train_hist(epoch_train_losses, results_dir, show=False, epoch_plot=True, save=True)
        #utils.show_train_val_hist(epoch_train_losses, val_losses, results_dir, show=False, save=True)
    np.save(os.path.join(results_dir,"epoch_avg_trainloss"), epoch_train_losses)
    np.save(os.path.join(results_dir,"epoch_val_loss"), val_losses)

if __name__ == '__main__':

    '''
    Main file for running the training. Initializes all the required variables and flags
    '''
    import os
    from PIL import Image

    # training hyperparameters
    batch_size = 64
    lr = 0.00001
    epochs = 20
    data_dir = 'data/3d'
    model_dir = 'model'
    checkpoint_dir = 'checkpoint'
    results_dir = '3d_results'
    print('learning rate', lr)
    print('epochs', epochs)
    #model parameters
    channels = 32
    vector_dim = 1
    restore = 'best'
    USE_GPU = True
    dtype = torch.float32
    print_every = 2    # iterations before printing
    # main_dir = os.path.relpath(os.path.dirname(__file__))
    # use GPU if available
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("ON GPU")
    else:
        device = torch.device('cpu')


    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if device == "cuda:0": torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Define the model and optimizer
    Dcnn_model = ThreeDcnn.ThreeDCNN(channels, vector_dim)
    optimizer = optim.Adam(Dcnn_model.parameters(), lr=lr, betas=(0.5, 0.999))

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
