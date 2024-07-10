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
from models.Fed import FedAvg 
import tenseal as ts

def encrypt_weights(weights, context):
    """Encrypt model weights using the provided TenSEAL context."""
    encrypted_weights = [ts.ckks_vector(context, weight.tolist()) for weight in weights]
    return encrypted_weights

def decrypt_weights(encrypted_weights, context):
    """Decrypt model weights."""
    decrypted_weights = [weight.decrypt() for weight in encrypted_weights]
    return np.array(decrypted_weights)

def he_avg_weights(encrypted_weights):
    """Homomorphically compute the average of encrypted weights."""
    avg_weights = encrypted_weights[0]
    for weights in encrypted_weights[1:]:
        avg_weights += weights
    avg_weights = avg_weights / len(encrypted_weights)
    return avg_weights

def sequential_process(args, text_widget, ax1, ax2, fig, canvas):
    def update_text(message):
        text_widget.insert(tk.END, message + '\n')
        text_widget.see(tk.END)  # Auto-scroll to the end

    def update_plots(epoch_losses, epoch_accuracies):
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

    # Setup TenSEAL context
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()

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

    update_text('Federated Learning Simulation started. Initializing model architecture...\n')
    update_text('Model architecture loaded and initialized. Starting training process on dataset: ' + args.dataset + '\n')
    net_glob.train()
    epoch_losses = []
    epoch_accuracies = []

    for iter in range(args.epochs):
        update_text(f'+++ Epoch {iter + 1} starts +++')
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        encrypted_weights = []
        local_losses = []

        for idx, user in enumerate(idxs_users):
            update_text(f'Starting training on client {user}')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # Encrypt weights here using CKKS
            ew = [ts.ckks_vector(context, w_i.numpy().flatten()) for k, w_i in w.items()]
            encrypted_weights.append(ew)
            update_text(f'Client {user} training complete. Loss: {loss:.4f}')
            update_text(f'Encrypting model weights at client {user}...')

            local_losses.append(loss)

        # Homomorphically average the encrypted weights
        update_text('Aggregating encrypted model updates...')
        avg_encrypted_weights = [sum(weights) / len(weights) for weights in zip(*encrypted_weights)]
        update_text('Decryption and aggregation of model updates at the server...')
        decrypted_avg_weights = {k: torch.tensor(w.decrypt()).view_as(net_glob.state_dict()[k]) for k, w in zip(net_glob.state_dict().keys(), avg_encrypted_weights)}

        net_glob.load_state_dict(decrypted_avg_weights)
        update_text('Decrypted aggregated model updates applied.')

        net_glob.eval()
        acc_train, _ = test_fun(net_glob, dataset_train, args)
        epoch_losses.append(np.mean(local_losses))
        epoch_accuracies.append(acc_train)

        update_plots(epoch_losses, epoch_accuracies)
        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}')
        update_text('---\n')

    exp_details(args)
    update_text('Training complete. Summary of results:')
    update_text(f'Final Training Accuracy: {acc_train:.2f}')


def create_gui(args):
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
