import logging
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import utils
import model.data_loader as data_loader
import model.data_loader_multichannel as data_loader_multi
import model.crnn as net
from evaluate import evaluate, evaluate_by_month, evaluate_single_value, evaluate_many
import torch.nn as nn
import os


def train(model, optimizer, loss_fn, dataloader):

    # set model to training mode
    model.train()

    #for m in model.modules():
     #   if isinstance(m, nn.BatchNorm2d):
      #      m.eval()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):

            model = model.to(device=device)  # move the model parameters to CPU/GPU
            model.train()  # put model to training mode

            train_batch = train_batch.to(device=device, dtype=dtype)
            labels_batch = labels_batch.to(device=device, dtype=dtype)

            # compute forward pass
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # in case you want to print the training loss after every n iterations
            #if i % print_every == 0:
             #   print('Iteration %d, loss = %.4f' % (i, loss.item()))

            t.update()


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs):

    best_val_MSE = float('inf')

    for epoch in range(epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, epochs))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader)

        # Evaluate MSE for one epoch on train and validation set
        train_MSE = evaluate(results_dir, model, nn.MSELoss(), train_dataloader, device, dtype)
        val_MSE = evaluate(results_dir, model, nn.MSELoss(), val_dataloader, device, dtype)
        #test_MSE = evaluate(results_dir, model, nn.MSELoss(), test_dataloader, device, dtype)
        # Evaluate L1 for one epoch on train and validation set
        train_L1 = evaluate(results_dir, model, nn.L1Loss(), train_dataloader, device, dtype)
        val_L1 = evaluate(results_dir, model, nn.L1Loss(), val_dataloader, device, dtype)

        scheduler.step(train_MSE)

        # save training history in csv file:
        utils.save_history(epoch, train_MSE, val_MSE, val_MSE, train_L1, val_L1, results_dir)

        # print losses
        logging.info("- Train average RMSE loss: " + str(np.sqrt(train_MSE)))
        logging.info("- Validation average RMSE loss: " + str(np.sqrt(val_MSE)))
        #logging.info("- Test average RMSE loss: " + str(np.sqrt(test_MSE)))

        # save MSE if is the best
        is_best = val_MSE <= best_val_MSE
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best evaluation loss")
            # Save best val loss in a txt file in the checkpoint directory
            best_val_path = "best_val_loss.txt"
            utils.save_dict_to_txt(val_MSE, results_dir, best_val_path, epoch)
            best_val_MSE = val_MSE

        # Save latest val metrics in a json file in the results directory
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
    old_checkpoint_dir = 'experiments_sst_output/GCM_ensemble_training/ensemble_6m_3models/results_6m_ensemble_3_models_drop0.2/checkpoint'
    checkpoint_dir = 'results_ensemble2_TL_conv/checkpoint'
    results_dir = 'results_ensemble2_TL_conv'
    dropout = None
    variables = ['nst']

    # choose model
    model_name = 'crnn'    # cnn / crnn / crnn_many

    # training hyperparameters
    batch_size = 1
    lr = 0.000001
    #lr_step_size = 15  # epochs to every decay lr
    #lr_factor = 0.2    # decay lr multiplicative factor
    epochs = 30
    channels = 10    # channels for the first cnn filter (this defines the number of filters in all the convolution layers)

    # hyperparameters for CRNN
    vector_dim = 500    # features vector dimension after CNN part of CRNN
    rnn_hidden_size = 500
    rnn_num_layers = 2

    # other parameters
    restore_file= 'best'
    transfer_learning = 'conv' # None, 'fc', 'conv', 'all'
    USE_GPU = True
    dtype = torch.float32

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
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    utils.set_logger(os.path.join(results_dir, 'train.log'))

    # print and save hyperparameters
    logging.info("learning rate: " + str(lr))
    logging.info("epochs: " + str(epochs))
    logging.info("batch size: " + str(batch_size))
    logging.info("transfer learning: " + str(transfer_learning))
    logging.info("Variables: " + str(variables))

    #Define the model, dataset
    if model_name == 'cnn':
        model = net.CNN(channels)
        data_dir = 'data/2d'
        logging.info("model: CNN")
    if model_name == 'crnn':
        model = net.CRNN(len(variables), channels, vector_dim, rnn_hidden_size, rnn_num_layers, dropout=dropout)
        data_dir = 'data/3d_multivar'
        #data_dir = '/Volumes/matiascastilloHD/CLIMATEAI/3d_ensemble2_6m'
        logging.info("model: CRNN")
        logging.info("data: CNRM-MPI")
        logging.info("encoding dimesion: " + str(vector_dim))
        logging.info("RNN layers: " + str(rnn_num_layers))
        logging.info("RNN hidden units: " + str(rnn_hidden_size))

    # initialize model weights
    model.apply(net.initialize_weights)

        # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(old_checkpoint_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model)
        logging.info("Parameters loaded")

    #path = 'data/individual/skt_from_2016-04_to_2018-03.npy'
    #evaluate_single_value(model, path, device, dtype)


    # define optimizer depending on transfer learning
    if transfer_learning == None or transfer_learning == 'all':
        params = model.parameters()
    if transfer_learning == 'fc':
        params = list(model.fc1.parameters()) + list(model.fc2.parameters())
    if transfer_learning == 'conv':
        params = list(model.conv6.parameters()) + list(model.conv5.parameters()) \
                 + list(model.fc1.parameters()) + list(model.fc2.parameters())

    optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    #define a learning rate decay by creating a pytorch scheduler

    #scheduler = optim.lr_scheduler.StepLR(optimizer, lr_step_size, gamma=lr_factor, last_epoch=-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, threshold=0.0001)

    # fetch loss function
    loss_fn = net.loss_fn

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    # fetch dataloaders

    if len(variables) > 1:
        dataloaders = data_loader_multi.fetch_dataloader(['train', 'val'], data_dir, batch_size, model_name, variables)
    else:
         dataloaders = data_loader.fetch_dataloader(['train', 'val', 'test'], data_dir, batch_size, model_name)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']
    logging.info("- done.")


    # call the function you want to use: evaluate, or train and evaluate.
    #train_MSE = evaluate(results_dir, model, nn.MSELoss(), train_dl, device, dtype)
    #val_MSE = evaluate(results_dir, model, nn.MSELoss(), val_dl, device, dtype)
    test_MSE = evaluate(results_dir, model, nn.MSELoss(), test_dl, device, dtype)
    #val_L1 = evaluate(model, nn.L1Loss(), val_dl, device, dtype)
    #test_L1 = evaluate(model, nn.L1Loss(), test_dl, device, dtype)
    #print('Training RMSE before training: ' + str(np.sqrt(train_MSE)))
    #print('Validation RMSE before training: ' + str(np.sqrt(val_MSE)))
    print('Test RMSE before training: ' + str(np.sqrt(test_MSE)))
    #print('Validation L1 before training: ' + str(val_L1))
    #print('Test L1 before training: ' + str(test_L1))

    #evaluate_single_value(model, nn.L1Loss(), test_dl, device, dtype)

    # Train the model
    #logging.info("Starting training for {} epoch(s)".format(epochs))
    #train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, epochs)



