'''
Sequential Federated Learning Algorithm THREADED WITH CKKS ENCRYPTION
Alian Haidar - 22900426
Last Modified: 2024-08-14
'''

import copy
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, font, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
import tenseal as ts

from models.Nets import MLP, Mnistcnn, Cifar10cnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser

# CKKS Context Setup
def create_ckks_context():
    '''Function to create a CKKS context with the specified parameters'''
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def update_text(message, text_widget):
    text_widget.insert(tk.END, message + '\n')
    text_widget.see(tk.END)

def check_for_nan(weights_dict, description, text_widget):
    '''Some context NaN values can appear, check for them'''
    for name, weight in weights_dict.items():
        if torch.isnan(weight).any():
            update_text(f"NaN detected in {description}: {name}", text_widget)
            return True
    return False

def log_weight_stats(weights_dict, description, text_widget):
    '''Log the mean, max, and min of the weights'''
    for name, weight in weights_dict.items():
        weight_np = weight.cpu().numpy()
        update_text(f"{description} - {name}: mean={weight_np.mean():.4f}, max={weight_np.max():.4f}, min={weight_np.min():.4f}", text_widget)

def encrypt_weights(weights_dict, context, text_widget, chunk_size=1000):
    '''Encrypt the weights using CKKS encryption'''
    update_text("Encrypting weights using CKKS encryption...", text_widget)
    encrypted_weights = {}
    for name, weight in weights_dict.items():
        weight_np = weight.cpu().numpy().astype(np.float64)
        if np.isnan(weight_np).any():
            update_text(f"NaN detected in {name} before encryption", text_widget)
            continue
        
        weight_chunks = np.array_split(weight_np.flatten(), max(1, len(weight_np.flatten()) // chunk_size))
        encrypted_chunks = [ts.ckks_vector(context, chunk) for chunk in weight_chunks]
        encrypted_weights[name] = encrypted_chunks
    
    update_text("Encryption completed.", text_widget)
    return encrypted_weights

def decrypt_weights(encrypted_weights, context, original_shapes, text_widget, client_count):
    '''Decrypt the weights using CKKS decryption'''
    update_text("Decrypting aggregated weights using CKKS decryption...", text_widget)
    decrypted_weights = {}
    for name, enc_weight_chunks in encrypted_weights.items(): # Iterate over each weight
        decrypted_flat = []
        for enc_weight in enc_weight_chunks:
            decrypted_flat.extend(enc_weight.decrypt()) # Decrypt each chunk and flatten

        # Apply the averaging by dividing by client_count during decryption
        decrypted_array = (np.array(decrypted_flat, dtype=np.float32) / client_count).reshape(original_shapes[name])
        decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)

        log_weight_stats({name: decrypted_weights[name]}, "Decrypted Weights", text_widget)

        if check_for_nan({name: decrypted_weights[name]}, "Decrypted Weights", text_widget):
            raise ValueError(f"NaN detected in decrypted weights: {name}")

    update_text("Decryption completed.", text_widget)
    return decrypted_weights


def aggregate_encrypted_weights(encrypted_weights1, encrypted_weights2, client_count, text_widget):
    '''Aggregate the encrypted weights using homomorphic addition'''
    update_text("Aggregating encrypted weights using homomorphic addition...", text_widget)
    aggregated_weights = {}
    for name in encrypted_weights1.keys():
        aggregated_chunks = []
        for enc_w1, enc_w2 in zip(encrypted_weights1[name], encrypted_weights2[name]):
            aggregated_chunk = enc_w1 + enc_w2  # Perform addition without scaling here
            aggregated_chunks.append(aggregated_chunk)
        aggregated_weights[name] = aggregated_chunks
    update_text("Encrypted weights aggregation completed.", text_widget)
    return aggregated_weights



def compare_weights(weights1, weights2, text_widget, client_count):
    '''Some epochs may return different values between decrypted and true values, compare the weights to see if match'''
    update_text("Comparing decrypted and original weights...", text_widget)
    for name in weights1.keys():
        # Adjust the original weights to account for the averaging
        adjusted_weights2 = weights2[name] / client_count
        match = torch.allclose(weights1[name], adjusted_weights2, atol=1)
        update_text(f"Comparison for {name}: {'Match' if match else 'Mismatch'}", text_widget)
    update_text("Comparison completed.", text_widget)


def client_training(client_id, dataset_train, dict_party_user, net_glob, text_widget, context, received_encrypted_weights=None):
    '''Function to train a client on its local dataset'''
    update_text(f'Starting training on client {client_id}', text_widget)

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[client_id]) # Initialize local training

    net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict())) # Load the global model

    local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) # Perform training

    if check_for_nan(local_weights, f"Client {client_id} local weights after training", text_widget): # Check for NaN values
        raise ValueError(f"NaN detected in Client {client_id} local weights after training.")

    update_text(f'Client {client_id} has completed training. Loss: {loss:.4f}', text_widget)

    log_weight_stats(local_weights, f"Client {client_id} local weights before encryption", text_widget) # Log the weights

    if check_for_nan(local_weights, f"Client {client_id} local weights before encryption", text_widget): # Check for NaN values
        raise ValueError(f"NaN detected in Client {client_id} local weights before encryption.")

    encrypted_weights = encrypt_weights(local_weights, context, text_widget) # Encrypt the weights

    if received_encrypted_weights is not None: # If there are already aggregated weights
        update_text(f'Client {client_id} is aggregating received encrypted weights.', text_widget)
        encrypted_weights = aggregate_encrypted_weights(encrypted_weights, received_encrypted_weights, text_widget)

    update_text(f'Client {client_id} has completed aggregation.', text_widget)
    return encrypted_weights, local_weights, loss  # Return both encrypted and unencrypted weights

def threaded_client_training(client_id, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results):
    '''Threaded function to train a client on its local dataset'''
    encrypted_weights, local_weights, loss = client_training(
        client_id,
        dataset_train,
        dict_party_user,
        net_glob,
        text_widget,
        context,
        received_encrypted_weights
    )
    results[client_id] = (encrypted_weights, local_weights, loss)

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    '''Sequential Federated Learning Algorithm with CKKS Encryption'''
    def update_plots(epoch_losses, epoch_accuracies):
        ax1.clear()
        ax2.clear()

        # Plotting with smaller font sizes
        ax1.plot(epoch_losses, label='Average Loss per Epoch', marker='o')
        ax1.set_title('Training Loss Over Epochs', fontsize=8)  # Smaller title
        ax1.set_xlabel('Epoch', fontsize=6)  # Smaller x-label
        ax1.set_ylabel('Loss', fontsize=6)  # Smaller y-label
        ax1.legend(fontsize=8)  # Smaller legend

        ax2.plot(epoch_accuracies, label='Accuracy per Epoch', marker='o')
        ax2.set_title('Training Accuracy Over Epochs', fontsize=8)  # Smaller title
        ax2.set_xlabel('Epoch', fontsize=6)  # Smaller x-label
        ax2.set_ylabel('Accuracy', fontsize=6)  # Smaller y-label
        ax2.legend(fontsize=8)  # Smaller legend

        # Adjust tick parameters for smaller size
        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)

        canvas.draw()


    dataset_train, dataset_test, dict_party_user, _ = get_dataset(args)

    if args.model == 'cnn':
        if args.dataset == 'MNIST':
            net_glob = Mnistcnn(args=args).to(args.device)
        elif args.dataset == 'CIFAR10':
            net_glob = Cifar10cnn(args=args).to(args.device)  # Assuming Cifar10cnn is your CNN model for CIFAR-10
        else:
            update_text('Error: unrecognized dataset for CNN model', text_widget)
            return
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

    context = create_ckks_context()

    net_glob.train() # Set the model to training mode

    epoch_losses = [] # Store the loss per epoch
    epoch_accuracies = [] # Store the accuracy per epoch

    for iter in range(args.epochs): # Number of global rounds
        update_text(f'+++ Epoch {iter + 1} starts +++', text_widget)
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        local_losses = []

        original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

        received_encrypted_weights = None
        received_unencrypted_weights = None  # Store unencrypted weights for comparison

        results = {}
        threads = []
        client_count = 0  # Initialize client count for averaging

        for idx, user in enumerate(idxs_users): # Iterate over each client
            thread = threading.Thread( # Create a thread for each client
                target=threaded_client_training,
                args=(user, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        for idx, user in enumerate(idxs_users): # Iterate over each client
            encrypted_weights, local_weights, loss = results[user]
            client_count += 1  # Increment the number of clients after each aggregation

            if received_encrypted_weights is not None:
                received_encrypted_weights = aggregate_encrypted_weights(
                    received_encrypted_weights,
                    encrypted_weights,
                    client_count,
                    text_widget
                )
            else:
                received_encrypted_weights = encrypted_weights  # First client, no aggregation needed

            received_unencrypted_weights = local_weights  # Keep track of unencrypted weights
            local_losses.append(loss)

        update_text('Final client sending aggregated encrypted weights to server.', text_widget)
        decrypted_weights = decrypt_weights(received_encrypted_weights, context, original_shapes, text_widget, client_count) # Decrypt the weights


        if check_for_nan(decrypted_weights, "Global Model Weights", text_widget):
            raise ValueError("NaN detected in global model weights before updating.")
        
        log_weight_stats(decrypted_weights, "Global Model Weights", text_widget)

        # Compare decrypted weights with unencrypted weights
        compare_weights(decrypted_weights, received_unencrypted_weights, text_widget, client_count)


        net_glob.load_state_dict(decrypted_weights)  # Encrypted weights used for comparison
        #net_glob.load_state_dict(received_unencrypted_weights)
        update_text('Server has updated the global model with final aggregated weights.', text_widget)

        net_glob.eval() # Set the model to evaluation mode
        acc_train, _ = test_fun(net_glob, dataset_train, args) # Test the model on the training data
        epoch_losses.append(np.mean(local_losses)) # Store the loss
        epoch_accuracies.append(acc_train) # Store the accuracy

        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
        update_plots(epoch_losses, epoch_accuracies)
        update_text('---\n', text_widget)

    exp_details(args) # Log the experimental details
    update_text('Training complete. Summary of results:', text_widget)
    update_text(f'Final Training Accuracy: {acc_train:.2f}', text_widget)


def create_gui(args):
    '''Create a GUI for the Federated Learning Simulation'''
    root = tk.Tk()
    root.title('Federated Learning Simulation with CKKS Encryption')

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

    thread = threading.Thread(target=run_learning_process)
    thread.start()

    root.mainloop()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    create_gui(args)
