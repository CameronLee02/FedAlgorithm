'''
Sequential Federated Learning Algorithm
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

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    '''Sequential Federated Learning Algorithm process for a given number of epochs.'''
    
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

    # Use for choosing model, currently supporrts MLP and CNN for (MNIST and Synthetic) datasets, will be changed.
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

    update_text('Federated Learning Simulation started. Initializing model architecture...\n')
    update_text('Model architecture loaded and initialized. Starting training process on dataset: ' + args.dataset + '\n')
    update_plots([], [])

    net_glob.train() # Set the model to training mode

    epoch_losses = []
    epoch_accuracies = []

    for iter in range(args.epochs): # Number of training epochs
        update_text(f'+++ Epoch {iter + 1} starts +++')
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        local_losses = []
        cumulative_weights = None

        for idx, user in enumerate(idxs_users): # Iterate through each client
            update_text(f'Starting training on client {user}')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user]) # Initialize model train per client own data
            local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) # Train the model
            local_losses.append(loss) # Calculate loss

            # Aggregating weights using FedAvg
            if cumulative_weights is None:
                cumulative_weights = local_weights # Initialize cumulative weights with the first client's weights
                update_text(f'First client {user} has completed training. No aggregation needed.')
            else:
                cumulative_weights = FedAvg([cumulative_weights, local_weights]) # Aggregate weights using FedAvg as there are more than one clients to append

            update_text(f'Client {user} has completed training. Loss: {loss:.4f}')
            if idx < len(idxs_users) - 1:
                next_client = idxs_users[idx + 1] # Get the next client using randomised index
                update_text(f'Passing aggregated weights from Client {user} to Client {next_client}')

        # Updating global model on the server with the final weights from the last client
        net_glob.load_state_dict(cumulative_weights)
        update_text('Server has updated the global model with final aggregated weights.')

        net_glob.eval() # Set the model to evaluation mode
        acc_train, _ = test_fun(net_glob, dataset_train, args) # Test the model on the training dataset
        epoch_losses.append(np.mean(local_losses))
        epoch_accuracies.append(acc_train)

        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}')
        update_plots(epoch_losses, epoch_accuracies)
        update_text('---\n')

    exp_details(args)
    update_text('Training complete. Summary of results:')
    update_text(f'Final Training Accuracy: {acc_train:.2f}')



def create_gui(args):
    '''Function to create the GUI for the simulation.'''

    root = tk.Tk()
    root.title('Federated Learning Simulation')

    # Create a scrolled text area widget
    custom_font = font.Font(family="San Francisco", size=16)
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=custom_font)
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Setup for Matplotlib figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


    # Function to run the learning process
    def run_learning_process():
        start_time = time.time()
        sequential_process(args, text_area, ax1, ax2, fig, canvas)
        total_time = time.time() - start_time
        text_area.insert(tk.END, f"Total time for completion: {total_time / 60:.2f} minutes.")

    # Start the learning process in a separate thread to keep GUI responsive
    thread = threading.Thread(target=run_learning_process)
    thread.start()

    root.mainloop()


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    create_gui(args)


'''
How to Run: 

python main_fed_sequential.py --dataset=MNIST --model=cnn --alpha=1 --num_users=6 --local_ep=5


'''