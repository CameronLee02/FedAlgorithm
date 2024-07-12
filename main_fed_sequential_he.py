import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, font
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tenseal as ts
import copy

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from models.Fed import FedAvg 

def setup_tenseal_context():
    """Set up TenSEAL context for CKKS."""
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def encrypt_weights(weights, context):
    """Encrypt model weights ensuring uniform vector size, adjust size as needed."""
    fixed_size = 4096  # Ensure this size fits your model size
    padded_weights = np.pad(weights, (0, max(0, fixed_size - len(weights))), 'constant', constant_values=(0))
    return ts.ckks_vector(context, padded_weights)

def decrypt_weights(encrypted_weights, context):
    """Decrypt model weights."""
    return np.array(encrypted_weights.decrypt())

def he_avg_weights(encrypted_weights, context):
    """Homomorphically compute the average of encrypted weights."""
    if not encrypted_weights:
        return None
    # Compute the number of weights
    n_weights = len(encrypted_weights)
    if n_weights == 0:
        return None 
    # Calculate the reciprocal of the number of weights
    reciprocal = 1.0 / n_weights  
    # Initialize the total with the first encrypted vector scaled by the reciprocal
    total = encrypted_weights[0] * reciprocal
    # Accumulate the rest of the weights scaled by the reciprocal
    for weights in encrypted_weights[1:]:
        total += weights * reciprocal
    return total


def sequential_process(args, text_widget, ax1, ax2, fig, canvas, context):
    def update_text(message):
        text_widget.insert(tk.END, message + '\n')
        text_widget.see(tk.END)

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

        encrypted_weights = []
        local_losses = []

        for idx, user in enumerate(idxs_users):
            update_text(f'Starting training on client {user}')
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[user])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

            # Encrypt weights
            weights = torch.nn.utils.parameters_to_vector(net_glob.parameters()).detach().numpy()
            encrypted_weights.append(encrypt_weights(weights, context))
            local_losses.append(loss)
            update_text(f'Client {user} has completed training. Loss: {loss:.4f}')

        # Homomorphically average the encrypted weights
        context = ...  # Ensure this is defined or available where needed
        avg_encrypted_weights = he_avg_weights(encrypted_weights, context)
        decrypted_avg_weights = decrypt_weights(avg_encrypted_weights, context)
        torch.nn.utils.vector_to_parameters(torch.tensor(decrypted_avg_weights), net_glob.parameters())
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
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=font.Font(family="San Francisco", size=16))
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    context = setup_tenseal_context()

    def run_learning_process():
        start_time = time.time()
        sequential_process(args, text_area, ax1, ax2, fig, canvas, context)
        total_time = time.time() - start_time
        text_area.insert(tk.END, f"Total time for completion: {total_time / 60:.2f} minutes.")

    threading.Thread(target=run_learning_process).start()
    root.mainloop()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    create_gui(args)
