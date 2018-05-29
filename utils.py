import json
import logging
import os
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np


def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_txt(loss_to_save, results_dir, path, epoch=None,iteration = None):

    if not os.path.exists(results_dir):
        print("Results Directory does not exist! Making directory {}".format(results_dir))
        os.mkdir(results_dir)
    if epoch:
        file = open(os.path.join(results_dir, path),'w')
        file.write('Epoch:' + str(epoch+1) + '. loss' + str(loss_to_save))
    if iteration:
        with open ("val_loss.txt","a") as f:
            f.write("Iteration {}, loss {} \n".format(iteration, loss_to_save))


    # with open(json_path, 'w') as f:
    #     # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
    #     d = {k: float(v) for k, v in d.items()}
    #     json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):

    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    if not state.get('iter'):
        torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

# save and show plot of epoch vs loss, or batches vs loss.
def show_train_hist(train_losses, results_dir, epoch_plot=True, show = False, save = False):

    if not os.path.exists(results_dir):
        print("Results Directory does not exist! Making directory {}".format(results_dir))
        os.mkdir(results_dir)

    x = range(len(train_losses))
    y = train_losses
    plt.plot(x, y, label='train_loss')
    plt.xlabel('Epoch') if epoch_plot else plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(results_dir, 'epoch_train_history')) if epoch_plot else plt.savefig(os.path.join(results_dir, 'batch_train_history'))
    if show:
        plt.show()
    else:
        plt.close()


# save and show plot of epoch vs loss for both train and validation sets.
def show_train_val_hist(train_losses, val_losses, results_dir, show=True, save=True):

    if not os.path.exists(results_dir):
        print("Results Directory does not exist! Making directory {}".format(results_dir))
        os.mkdir(results_dir)

    x = range(len(train_losses))
    print(x)
    y1 = train_losses
    y2 = val_losses
    plt.plot(x, y1, label='train_loss')
    plt.plot(x, y2, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(results_dir, 'train_val_history'))
    if show:
        plt.show()
    else:
        plt.close()
