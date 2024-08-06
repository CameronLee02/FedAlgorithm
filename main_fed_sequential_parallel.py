'''
Sequential Federated Learning Algorithm with Parallel Training
Alian Haidar - 22900426
Last Modified: 2024-07-24
'''

import copy
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, ttk, font
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from models.Fed import FedAvg  # Import FedAvg function from models/Fed.py

def train_model(client_data, net, return_dict, client_id, lock):
    '''Train model for a single client and save the result in a shared dictionary.'''
    net_local, loss = client_data.train(net=copy.deepcopy(net))  # Make sure to pass a deep copy of the net
    with lock:  # Use a lock to synchronize access to the shared dictionary
        return_dict[client_id] = (net_local.state_dict(), loss)  # Store state_dict instead of the model itself

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    '''Adjustments for parallel training and synchronization.'''

    def update_text(message):
        '''Function to update the text area in the GUI.'''
        text_widget.insert(tk.END, message + '\n')
        text_widget.see(tk.END)

    def update_plots(epoch_losses, epoch_accuracies):
        '''Function to update the plots in the GUI.'''
        ax1.clear()
        ax2.clear()
        ax1.plot(epoch_losses, label='Average Loss per Epoch', marker='o')
        ax1.set_title('Training Loss Over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epoch_accuracies, label='Accuracy per Epoch', marker='o')
        ax2.set_title('Training Accuracy Over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        canvas.draw()

    dataset_train, dataset_test, dict_party_user, _ = get_dataset(args)
    if args.model == 'cnn' and args.dataset == 'MNIST':
        net_glob = Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for dim in dataset_train[0][0].shape:
            len_in *= dim
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        update_text('Error: unrecognized model')
        return
    
    net_glob.train()
    epoch_losses = []
    epoch_accuracies = []

    for iter in range(args.epochs):
        update_text(f'+++ Epoch {iter + 1} starts +++')
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        threads = []
        local_results = {}
        lock = threading.Lock()  # Create a lock for thread synchronization

        # Start threads for each client
        for user in idxs_users:
            client_data = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            thread = threading.Thread(target=train_model, args=(client_data, net_glob, local_results, user, lock))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Aggregate weights after all threads have finished
        cumulative_weights = None
        for user in idxs_users:
            local_weights, loss = local_results[user]
            if cumulative_weights is None:
                cumulative_weights = local_weights
            else:
                cumulative_weights = FedAvg([cumulative_weights, local_weights])

        net_glob.load_state_dict(cumulative_weights)
        update_text('Server has updated the global model with final aggregated weights.')

        net_glob.eval()
        acc_train, _ = test_fun(net_glob, dataset_train, args)
        epoch_losses.append(np.mean([local_results[user][1] for user in idxs_users]))
        epoch_accuracies.append(acc_train)

        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}')
        update_plots(epoch_losses, epoch_accuracies)
        update_text('---\n')

    exp_details(args)
    update_text('Training complete. Summary of results:')
    update_text(f'Final Training Accuracy: {acc_train:.2f}')


def create_gui(args):
    '''Create the GUI for the simulation.'''
    root = tk.Tk()
    root.title('Federated Learning Simulation')

    custom_font = font.Font(family="San Francisco", size=16)
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=custom_font)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def run_learning_process():
        start_time = time.time()
        sequential_process(args, text_area, ax1, ax2, fig, canvas)
        total_time = time.time() - start_time
        text_area.insert(tk.END, f"Total time for completion: {total_time / 60:.2f} minutes.")

    threading.Thread(target=run_learning_process).start()

    root.mainloop()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    create_gui(args)
