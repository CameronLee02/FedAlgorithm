import copy
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from phe import paillier

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

def generate_paillier_keypair():
    """Generate a Paillier key pair."""
    public_key, private_key = paillier.generate_paillier_keypair()
    return public_key, private_key

def encrypt_weight(weight, public_key):
    """Encrypt the weight using Paillier public key."""
    weight_flat = weight.cpu().numpy().flatten().astype(np.float64)  # Ensure float64 for compatibility
    if np.any(np.isnan(weight_flat)):
        raise ValueError("Weights contain NaN values.")
    encrypted_weight = [public_key.encrypt(x) for x in weight_flat]
    return encrypted_weight

def decrypt_weight(encrypted_weight, private_key, original_shape):
    """Decrypt the weight using Paillier private key."""
    decrypted_flat = [private_key.decrypt(x) for x in encrypted_weight]
    decrypted_array = np.array(decrypted_flat).reshape(original_shape)
    return torch.from_numpy(decrypted_array).float()  # Convert back to float32

def encrypt_edge_weights(weights_dict, public_key):
    """Encrypt specific edge weights for a single client."""
    encrypted_weights = {}
    for name, weight in weights_dict.items():
        if name in ['conv1.weight', 'fc3.weight']:  # Only encrypt specific edge weights
            encrypted_weights[name] = encrypt_weight(weight, public_key)
        else:
            encrypted_weights[name] = weight  # Keep other weights unchanged
    return encrypted_weights

def aggregate_encrypted_weights(current_encrypted, new_encrypted):
    """Aggregate encrypted weights from the current and new updates."""
    aggregated_weights = {}
    for name in current_encrypted.keys():
        if isinstance(current_encrypted[name], list):  # Only aggregate encrypted weights
            aggregated_weights[name] = [x + y for x, y in zip(current_encrypted[name], new_encrypted[name])]
        else:
            aggregated_weights[name] = current_encrypted[name] + new_encrypted[name]
    return aggregated_weights

def decrypt_edge_weights(encrypted_weights, private_key, original_shapes):
    """Decrypt aggregated encrypted weights using the shared private key."""
    decrypted_weights = {}
    for name, encrypted_data in encrypted_weights.items():
        if isinstance(encrypted_data, list):
            decrypted_weights[name] = decrypt_weight(encrypted_data, private_key, original_shapes[name])
        else:
            decrypted_weights[name] = encrypted_data
    return decrypted_weights

def FedAvg(weights_list):
    """Federated Averaging for model weights."""
    w_avg = copy.deepcopy(weights_list[0])
    for k in w_avg.keys():
        if isinstance(w_avg[k], list):  # Skip encrypted lists
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

    net_glob.train()  # Set the model to training mode

    epoch_losses = []
    epoch_accuracies = []

    # Generate a shared Paillier key pair for all clients
    shared_public_key, shared_private_key = generate_paillier_keypair()

    for iter in range(args.epochs):  # Number of training epochs
        update_text(f'+++ Epoch {iter + 1} starts +++')
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        local_losses = []

        # Initialize the cumulative encrypted weights with the server's global model
        cumulative_encrypted_weights = encrypt_edge_weights(net_glob.state_dict(), shared_public_key)

        for idx, user in enumerate(idxs_users):  # Iterate through each client
            update_text(f'Starting training on client {user}')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            local_losses.append(loss)

            # Encrypt edge weights for this client using the shared public key
            encrypted_weights = encrypt_edge_weights(local_weights, shared_public_key)

            # Aggregate current client weights with cumulative weights
            cumulative_encrypted_weights = aggregate_encrypted_weights(cumulative_encrypted_weights, encrypted_weights)

            update_text(f'Client {user} has completed training. Loss: {loss:.4f}')
            if idx < len(idxs_users) - 1:
                next_client = idxs_users[idx + 1]
                update_text(f'Passing aggregated weights from Client {user} to Client {next_client}')

        # Last client sends the aggregated weights back to the server
        update_text('Last client has sent the aggregated weights back to the server.')

        # Decrypt aggregated weights with the shared private key
        decrypted_weights = decrypt_edge_weights(cumulative_encrypted_weights, shared_private_key, {name: local_weights[name].shape for name in local_weights})

        # Update global model
        net_glob.load_state_dict(decrypted_weights)
        update_text('Server has updated the global model with final aggregated weights.')

        net_glob.eval()  # Set the model to evaluation mode
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
