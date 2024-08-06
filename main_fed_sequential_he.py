'''
Sequential Federated Learning Algorithm with Paillier Encryption
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
from phe import paillier

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from models.Fed import FedAvg  # Import FedAvg function from models/Fed.py

def choose_model(args, dataset_train):
    '''Initialize and return the appropriate model based on args.'''
    if args.model == 'cnn' and args.dataset == 'MNIST':
        len_in = 1
        for dim in dataset_train[0][0].shape:
            len_in *= dim
        return Mnistcnn(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for dim in dataset_train[0][0].shape:
            len_in *= dim
        return MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError('Error: unrecognized model')

def encrypt_weights(weights, public_key):
    '''Encrypt model weights.'''
    encrypted_weights = {}
    for key, val in weights.items():
        print("Encrypting...")
        # Convert tensor to list of floats, then encrypt each element
        flat_list = val.flatten().tolist()  # Convert tensor to list of Python floats
        encrypted_weights[key] = [public_key.encrypt(v) for v in flat_list]
    return encrypted_weights



def decrypt_weights(encrypted_weights, private_key):
    '''Decrypt model weights.'''
    decrypted_weights = {}
    for key, enc_vals in encrypted_weights.items():
        decrypted_weights[key] = torch.tensor([private_key.decrypt(v) for v in enc_vals])
    return decrypted_weights

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    '''Sequential Federated Learning Algorithm process for a given number of epochs.'''
    # Generate Paillier keypair
    public_key, private_key = paillier.generate_paillier_keypair()

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
    net_glob = choose_model(args, dataset_train)

    update_text('Federated Learning Simulation started. Initializing model architecture...\n')
    update_text('Model architecture loaded and initialized. Starting training process on dataset: ' + args.dataset + '\n')
    update_plots([], [])
    net_glob.train()

    epoch_losses = []
    epoch_accuracies = []

    for iter in range(args.epochs):
        update_text(f'+++ Epoch {iter + 1} starts +++')
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        local_losses = []
        encrypted_cumulative_weights = None

        for idx, user in enumerate(idxs_users):
            update_text(f'Starting training on client {user}')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_losses.append(loss)
            
            encrypted_local_weights = encrypt_weights(local_weights, public_key)
            
            if encrypted_cumulative_weights is None:
                encrypted_cumulative_weights = encrypted_local_weights
            else:
                encrypted_cumulative_weights = FedAvg([encrypted_cumulative_weights, encrypted_local_weights], public_key)

            update_text(f'Client {user} has completed training. Loss: {loss:.4f}')
            if idx < len(idxs_users) - 1:
                next_client = idxs_users[idx + 1]
                update_text(f'Passing encrypted aggregated weights from Client {user} to Client {next_client}')

        decrypted_final_weights = decrypt_weights(encrypted_cumulative_weights, private_key)
        net_glob.load_state_dict(decrypted_final_weights)
        update_text('Server has updated the global model with decrypted aggregated weights.')

        net_glob.eval()
        acc_train, _ = test_fun(net_glob, dataset_train, args)
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
