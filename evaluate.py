
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import csv
import pandas as pd
from tqdm import tqdm

def evaluate(results_dir, model, loss, dataloader, device, dtype):
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    # summary for current eval loop
    eval_losses = []
    outputs = []
    labels = []
    # compute metrics over the dataset
    with torch.no_grad():
        with tqdm(total=len(dataloader)) as t:
            for val_batch, labels_val_batch in dataloader:
                val_batch = val_batch.to(device=device, dtype=dtype)
                labels_val_batch = labels_val_batch.to(device=device, dtype=dtype)
                # compute model output
                output_val = model(val_batch)
                outputs.append(output_val.item())
                labels.append(labels_val_batch.item())
                error = loss(output_val, labels_val_batch)
                eval_losses.append(error)
                t.update()
            outputfile = open('output.txt', 'w')
            for item in outputs:
                outputfile.write("%s\n" % item)
            correlation = np.corrcoef(np.asarray(outputs), np.asarray(labels))
            print('correlation: ', correlation)
            loss_avg = np.mean(eval_losses)

    return loss_avg

def evaluate_crossval(counter, current_pos, epoch, results_dir, model, loss, dataloader, device, dtype):
    model = model.to(device=device)
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
            error = loss(output_val, labels_val_batch)
            eval_losses.append(error)
            labels = labels_val_batch
        loss_avg = np.mean(eval_losses)
    outputs = np.asarray(output_val)
    labels = np.asarray(labels)

    crossval_month_path = os.path.join(results_dir,'crossval_month_results.csv')
    if (current_pos == 0 or current_pos == 35) and epoch == 0:
        crossval_month_df = pd.DataFrame(columns=['year', 'epoch','jan','feb','mar','apr', 'may','jun','jul','aug','sep','oct','nov','dec',
                                                  'ljan','lfeb','lmar','lapr', 'lmay','ljun','ljul','laug','lsep','loct','lnov','ldec'])
    else:
        crossval_month_df = pd.read_csv(crossval_month_path)
    crossval_month_df.loc[counter] = [current_pos+1, epoch, outputs[0], outputs[1], outputs[2],\
                                          outputs[3], outputs[4], outputs[5], outputs[6],\
                                          outputs[7], outputs[8], outputs[9], outputs[10],\
                                          outputs[11], labels[0], labels[1], labels[2], labels[3], labels[4], labels[5], labels[6],\
                                          labels[7], labels[8], labels[9], labels[10], labels[11]]
    crossval_month_df.to_csv(crossval_month_path, index=False)
    counter += 1

    return loss_avg, counter




def evaluate_by_month(model, loss, dataloader, device, dtype):
    months = dataloader.dataset.months.astype(int)
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    eval_losses = []
    monthly_losses = [0]*12
    months_count = [0]*12
    # compute metrics over the dataset
    with torch.no_grad():
        for val_batch, labels_val_batch in dataloader:
            val_batch = val_batch.to(device=device, dtype=dtype)
            labels_val_batch = labels_val_batch.to(device=device, dtype=dtype)
            # compute model output
            output_val = model(val_batch)
            error = loss(output_val, labels_val_batch)
            eval_losses.append(error)
    eval_losses = np.asarray(torch.cat(eval_losses,0))
    # take the monthly mean of the losses
    for i in range(months.shape[0]):
        monthly_losses[months[i]-1] += eval_losses[i]
        months_count[months[i]-1] += 1
    for i in range(12):
        monthly_losses[i] /= months_count[i]
    monthly_losses = np.sqrt(monthly_losses)
    print(monthly_losses)


def evaluate_single_value(model, path, device, dtype):
    model = model.to(device=device)
    model.eval()  # set model to evaluation mode
    # compute model output
    input = torch.FloatTensor(np.expand_dims(np.expand_dims(np.load(path), axis=0), axis=0))
    output = model(input)
    print(output.item())


def evaluate_many(model, loss, dataloader, device, dtype):
    eval_outputs = []
    eval_labels = []
    with torch.no_grad():
        for val_batch, labels_val_batch in dataloader:
            val_batch = val_batch.to(device=device, dtype=dtype)
            labels_val_batch = labels_val_batch.to(device=device, dtype=dtype)
            # compute model output
            output_val = model(val_batch)
            eval_outputs.append(output_val)
            eval_labels.append(labels_val_batch)
    outputs = np.asarray(torch.cat(eval_outputs,0))
    labels = np.asarray(torch.cat(eval_labels,0))
    L1_array = np.absolute(outputs - labels)
    L1_avg = np.mean(L1_array, axis=0)
    print('L1', L1_avg)
    MSE = np.mean(np.square(L1_array), axis=0)
    RMSE = np.sqrt(MSE)
    print(RMSE)
    print(np.mean(RMSE))

