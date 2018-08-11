import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# dataframes for each experiment
experiments = []
names = []

experiments.append(pd.read_csv('gcloud/results_20vector/loss_history.csv'))
names.append('encoding=20')
experiments.append(pd.read_csv('gcloud/results_500vector/loss_history.csv'))
names.append('encoding=500')
experiments.append(pd.read_csv('gcloud/results_1000vector/loss_history.csv'))
names.append('encoding=1000')
experiments.append(pd.read_csv('gcloud/results_2000vector/loss_history.csv'))
names.append('encoding=2000')
experiments.append(pd.read_csv('gcloud/results_500vector_channels15/loss_history.csv'))
names.append('encoding=500, channels=15')
experiments.append(pd.read_csv('gcloud/results_48m/loss_history.csv'))
names.append('encoding=500, 48months')
experiments.append(pd.read_csv('gcloud/results_48m_drop0.2/loss_history.csv'))
names.append('encoding=500, 48months, dropout=0.2')
experiments.append(pd.read_csv('gcloud/results_3months/loss_history.csv'))
names.append('encoding=500, 3m_ahead')
experiments.append(pd.read_csv('gcloud/results_3months_drop0.2/loss_history.csv'))
names.append('encoding=500, 3m_ahead, dropout=0.2')
experiments.append(pd.read_csv('gcloud/results_48m_6m/loss_history.csv'))
names.append('encoding=500, 6m_ahead')

def plot(error_type, experiments, names):

    for i, experiment in enumerate(experiments):
        if i == 0:
            ax = experiment.plot(x='epoch', y=error_type, label=error_type +  ', ' + names[i])
            plt.yticks(np.arange(0, 0.6, step=0.05))
            plt.xticks(np.arange(0, 30, step=5))
            plt.xlabel('Epoch')
        else:
            experiment.plot(ax=ax, y=error_type, label=error_type +  ', ' + names[i])
    #add grid lines
    ax.yaxis.grid(color='gray', linestyle='dashed')

plt.plot(x1, train_1, 'b', label='lr=2e-4 train')
plt.plot(x1, val_1, 'b--',label='lr=2e-4 val')
plt.plot(x2, train_2, 'g', label='lr=5e-5 train')
plt.plot(x2, val_2, 'g--', label='lr=5e-5 val')
plt.plot(x3, train_3,'r', label='lr=1e-5 train')
plt.plot(x3, val_3, 'r--' ,label='lr=1e-5 val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc=1)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("plots", 'train_val_plots'))


#plot('trainMSE', experiments[0:5], names[0:5])
#plot('valMSE', experiments[0:5], names[0:5])


experiments_plot2 = [experiments[i] for i in [1,5,6]]
names_plot2 = [names[i] for i in [1,5,6]]

#plot('trainMSE', experiments_plot2, names_plot2)
#plot('valMSE', experiments_plot2, names_plot2)

experiments_plot3 = [experiments[i] for i in [1,7,8,9]]
names_plot3 = [names[i] for i in [1,7,8,9]]
plot('trainMSE', experiments_plot3, names_plot3)
plot('valMSE', experiments_plot3, names_plot3)

plt.show()



'''
x = range(len(train_losses))
y = train_losses
plt.plot(x, y, label='train_loss')
 if epoch_plot else plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend(loc=4)
plt.grid(True)
plt.tight_layout()

'''



#if save:
 #   plt.savefig(os.path.join(results_dir, 'epoch_train_history')) if epoch_plot else plt.savefig(os.path.join(results_dir, 'batch_train_history'))
#if show:
 #   plt.show()
#else:
  #  plt.close()
