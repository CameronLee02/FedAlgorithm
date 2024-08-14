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
from concurrent.futures import ThreadPoolExecutor

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

# Paillier Key Generation
public_key, private_key = paillier.generate_paillier_keypair()

def update_text(message, text_widget):
    '''Function to update the text area in the GUI.'''
    text_widget.insert(tk.END, message + '\n')
    text_widget.see(tk.END)

def encrypt_weights(weights_dict, public_key, text_widget, chunk_size=500):
    """Encrypt weights using Paillier encryption with optimized parallel processing."""
    update_text("Encrypting weights using Paillier encryption...", text_widget)
    encrypted_weights = {}

    def encrypt_chunk(chunk):
        return [public_key.encrypt(float(x)) for x in chunk]

    with ThreadPoolExecutor() as executor:
        for name, weight in weights_dict.items():
            weight_np = weight.cpu().numpy().astype(np.float64)  # Convert to NumPy array and float64
            flattened_weights = weight_np.flatten()

            # Split weights into chunks for parallel processing
            start_time = time.time()
            encrypted_chunks = list(executor.map(
                encrypt_chunk, 
                (flattened_weights[i:i + chunk_size] for i in range(0, len(flattened_weights), chunk_size))
            ))
            end_time = time.time()
            update_text(f"Time taken to encrypt {name}: {end_time - start_time:.2f} seconds", text_widget)

            # Flatten the list of lists into a single list
            encrypted_weights[name] = [item for sublist in encrypted_chunks for item in sublist]

    update_text("Weights encrypted successfully.", text_widget)
    return encrypted_weights

def decrypt_weights(encrypted_weights, private_key, original_shapes, text_widget):
    """Decrypt weights using Paillier encryption."""
    update_text("Decrypting aggregated weights using Paillier decryption...", text_widget)
    decrypted_weights = {}
    for name, enc_weight in encrypted_weights.items():
        decrypted_flat = [private_key.decrypt(x) for x in enc_weight]
        decrypted_array = np.array(decrypted_flat).reshape(original_shapes[name])
        decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)
    update_text("Weights decrypted successfully.", text_widget)
    return decrypted_weights

def aggregate_encrypted_weights(encrypted_weights1, encrypted_weights2, text_widget):
    """Aggregate two sets of encrypted weights using homomorphic addition."""
    update_text("Aggregating encrypted weights using homomorphic addition...", text_widget)
    aggregated_weights = {}
    for name in encrypted_weights1.keys():
        aggregated_weights[name] = [w1 + w2 for w1, w2 in zip(encrypted_weights1[name], encrypted_weights2[name])]
    update_text("Encrypted weights aggregated successfully.", text_widget)
    return aggregated_weights

def client_training(client_id, dataset_train, dict_party_user, net_glob, text_widget, public_key, received_encrypted_weights=None):
    """Function for training a client."""
    update_text(f'Starting training on client {client_id}', text_widget)
    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[client_id])

    # Perform local training
    local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
    update_text(f'Client {client_id} has completed training. Loss: {loss:.4f}', text_widget)

    # Encrypt the local weights
    encrypted_weights = encrypt_weights(local_weights, public_key, text_widget)

    # If received weights, perform aggregation
    if received_encrypted_weights is not None:
        update_text(f'Client {client_id} is aggregating received encrypted weights.', text_widget)
        encrypted_weights = aggregate_encrypted_weights(encrypted_weights, received_encrypted_weights, text_widget)

    update_text(f'Client {client_id} has completed aggregation.', text_widget)
    return encrypted_weights, loss

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    '''Sequential Federated Learning Algorithm process for a given number of epochs.'''

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
        update_text('Error: unrecognized model', text_widget)
        return

    update_text('Federated Learning Simulation started. Initializing model architecture...\n', text_widget)
    update_text('Model architecture loaded and initialized. Starting training process on dataset: ' + args.dataset + '\n', text_widget)
    update_plots([], [])

    net_glob.train() # Set the model to training mode

    epoch_losses = []
    epoch_accuracies = []

    for iter in range(args.epochs): # Number of training epochs
        update_text(f'+++ Epoch {iter + 1} starts +++', text_widget)
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        local_losses = []

        # Store the original shapes for later decryption
        original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

        # Initialize received weights as None
        received_encrypted_weights = None

        for idx, user in enumerate(idxs_users):  # Iterate through each client
            update_text(f'Client {user} is starting training and encryption.', text_widget)
            encrypted_weights, loss = client_training(
                user,
                dataset_train,
                dict_party_user,
                net_glob,
                text_widget,
                public_key,
                received_encrypted_weights
            )
            received_encrypted_weights = encrypted_weights
            local_losses.append(loss)

        # The final client sends aggregated encrypted weights to the server for decryption
        update_text('Final client sending aggregated encrypted weights to server.', text_widget)
        decrypted_weights = decrypt_weights(received_encrypted_weights, private_key, original_shapes, text_widget)
        net_glob.load_state_dict(decrypted_weights)
        update_text('Server has updated the global model with final aggregated weights.', text_widget)

        net_glob.eval() # Set the model to evaluation mode
        acc_train, _ = test_fun(net_glob, dataset_train, args)
        epoch_losses.append(np.mean(local_losses))
        epoch_accuracies.append(acc_train)

        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
        update_plots(epoch_losses, epoch_accuracies)
        update_text('---\n', text_widget)

    exp_details(args)
    update_text('Training complete. Summary of results:', text_widget)
    update_text(f'Final Training Accuracy: {acc_train:.2f}', text_widget)


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

