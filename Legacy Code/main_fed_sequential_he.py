'''
RSA ENCRYPTION BUT ONLY THE BEGINNING AND END OF SERIES LINK
'''

import copy
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

# RSA Key Generation
key = RSA.generate(2048)
private_key = key.export_key()
public_key = key.publickey().export_key()

# Create cipher objects for encryption and decryption
encrypt_cipher = PKCS1_OAEP.new(RSA.import_key(public_key))
decrypt_cipher = PKCS1_OAEP.new(RSA.import_key(private_key))

def encrypt_edge_weights(weights_dict, cipher, max_chunk_size=190):
    """Encrypt specific edge weights using RSA with chunking."""
    encrypted_weights = {}
    for name, weight in weights_dict.items():
        if name in ['conv1.weight', 'fc3.weight']:  # Encrypt input and output layer weights
            weight_bytes = weight.cpu().numpy().tobytes()
            encrypted_chunks = [cipher.encrypt(weight_bytes[i:i + max_chunk_size]) 
                                for i in range(0, len(weight_bytes), max_chunk_size)]
            encrypted_weights[name] = encrypted_chunks
        else:
            encrypted_weights[name] = weight  # Keep other weights unchanged
    return encrypted_weights

def decrypt_edge_weights(encrypted_weights, cipher, original_shapes):
    """Decrypt specific edge weights using RSA with chunking."""
    decrypted_weights = {}
    for name, weight in encrypted_weights.items():
        if isinstance(weight, list):  # Check if the weight is a list of encrypted chunks
            decrypted_bytes = b''.join([cipher.decrypt(chunk) for chunk in weight])
            decrypted_array = np.frombuffer(decrypted_bytes, dtype=np.float32).reshape(original_shapes[name])
            decrypted_weights[name] = torch.from_numpy(decrypted_array)
        else:
            decrypted_weights[name] = weight
    return decrypted_weights

def FedAvg(weights_list):
    """Federated Averaging for model weights."""
    w_avg = copy.deepcopy(weights_list[0])
    for k in w_avg.keys():
        if isinstance(w_avg[k], list):  # Handle encrypted lists separately
            continue
        for i in range(1, len(weights_list)):
            w_avg[k] += weights_list[i][k]
        w_avg[k] = torch.div(w_avg[k], len(weights_list))
    return w_avg

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

    # Use for choosing model, currently supports MLP and CNN for (MNIST and Synthetic) datasets, will be changed.
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
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_losses.append(loss)

            # Save original shapes for reconstruction after decryption
            original_shapes = {name: weight.shape for name, weight in local_weights.items()}

            # Encrypt edge weights
            encrypted_weights = encrypt_edge_weights(local_weights, encrypt_cipher)

            if cumulative_weights is None:
                cumulative_weights = encrypted_weights
                update_text(f'First client {user} has completed training. No aggregation needed.')
            else:
                # Aggregate weights normally, skipping encryption here
                cumulative_weights = FedAvg([cumulative_weights, local_weights])

            update_text(f'Client {user} has completed training. Loss: {loss:.4f}')

        # Decrypt the edge weights after aggregation
        decrypted_weights = decrypt_edge_weights(cumulative_weights, decrypt_cipher, original_shapes)
        net_glob.load_state_dict(decrypted_weights)
        update_text('Server has updated the global model with final aggregated weights.')

        net_glob.eval() # Set the model to evaluation mode
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


'''
How to Run: 

python main_fed_sequential.py --dataset=MNIST --model=cnn --alpha=1 --num_users=6 --local_ep=5

'''
