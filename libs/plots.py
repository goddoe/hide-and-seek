import os
import time
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
    
def plot_loss_vs_epoch_and_save(report_dict, save_dir_path):
    fig, axs = plt.subplots(1, 2,figsize=(12,6))
    axs[0].plot([ loss for loss in report_dict['train_loss']], color='blue', label='train_loss')
    axs[0].plot([  loss for loss in report_dict['valid_loss']], color='red', label='valid_loss')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss') 
    axs[0].legend(loc='best', frameon=False)

    axs[1].plot([ accuracy for accuracy in report_dict['train_accuracy']], color='blue', label='train_accuracy')
    axs[1].plot([ accuracy for accuracy in report_dict['valid_accuracy']], color='red', label='valid_accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy') 
    axs[1].legend(loc='best', frameon=False)
    save_path = save_dir_path    
    try:
        os.makedirs(save_path)
    except:
        pass
    with open("{}/error_vs_loss.pickle".format(save_path), "wb") as f:
        pickle.dump(report_dict, f)
    fig.savefig("{}/error_vs_loss.png".format(save_path))
    return fig, axs



def plot_accuracy_validation_curve(report_list, x_label, save_dir_path=None):
    fig, axs = plt.subplots(1, 1,figsize=(6,6))
    axs.plot([report['train_score'] for report in report_list], color='blue', label='train_accuracy')
    axs.plot([ report['valid_score'] for report in report_list], color='red', label='valid_accuracy')
    axs.set_xlabel(x_label)
    axs.set_ylabel('accuracy') 
    axs.legend(loc='best', frameon=False)
    save_path = save_dir_path    
    if save_dir_path:
        try:
            os.makedirs(save_path)
        except:
            pass
        with open("{}/{}.pickle".format(save_path, x_label), "wb") as f:
            pickle.dump(report_list, f)
        fig.savefig("{}/{}.png".format(save_path, x_label))
    return fig, axs


