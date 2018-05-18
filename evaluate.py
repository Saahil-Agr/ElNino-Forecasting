
import logging
import os

import numpy as np
import torch
import torch.nn as nn

def evaluate(model, loss_fn, dataloader, device, dtype):

    model.eval()  # set model to evaluation mode

    # summary for current eval loop
    eval_losses = []

    # compute metrics over the dataset
    with torch.no_grad():
        for val_batch, labels_val_batch in dataloader:

            val_batch = val_batch.to(device=device, dtype=dtype)
            labels_val_batch = labels_val_batch.to(device=device, dtype=dtype)

            # compute model output
            output_val = model(val_batch)
            criterion = nn.MSELoss()
            loss = criterion(output_val, labels_val_batch)

            eval_losses.append(loss.item())

        loss_avg = np.mean(eval_losses)


    return loss_avg
