'''
Sequential Federated Learning Algorithm
Alian Haidar - 22900426
Last Modified: CODE INCORRECT, DO NOT USE
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
import tenseal as ts

from models.Nets import MLP, Mnistcnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser
from models.Fed import FedAvg  # Import the FedAvg function

def setup_he_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    return context

def encrypt_weights(weights, context):
    encrypted_weights = {}
    for key, weight in weights.items():
        encrypted_weights[key] = ts.ckks_vector(context, weight.numpy().flatten())
    return encrypted_weights

def decrypt_weights(encrypted_weights, context, shape_map):
    decrypted_weights = {}
    for key, encrypted_weight in encrypted_weights.items():
        decrypted_weights[key] = torch.tensor(encrypted_weight.decrypt()).view(shape_map[key])
    return decrypted_weights

def train_and_encrypt(data, net, context):
    local = LocalUpdate(args=args, dataset=data.dataset, idxs=data.idxs)
    local_weights, loss = local.train(net=copy.deepcopy(net).to(args.device))
    encrypted_weights = encrypt_weights(local_weights, context)
    return encrypted_weights, loss

def aggregate_encrypted_weights(encrypted_weights_list):
    aggregated_weights = encrypted_weights_list[0]
    for weights in encrypted_weights_list[1:]:
        for key in aggregated_weights:
            aggregated_weights[key] += weights[key]
    return aggregated_weights

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
    net_glob = MLP(dim_in=784, dim_hidden=200, dim_out=10).to(args.device)  # Example for MNIST

    shape_map = {key: val.shape for key, val in net_glob.state_dict().items()}  # Save shapes for decryption

    for iter in range(args.epochs):
        encrypted_aggregated_weights = None

        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)

        for idx, user in enumerate(idxs_users):
            encrypted_weights, loss = train_and_encrypt(dict_party_user[user], net_glob, context)
            if encrypted_aggregated_weights is None:
                encrypted_aggregated_weights = encrypted_weights
            else:
                encrypted_aggregated_weights = aggregate_encrypted_weights([encrypted_aggregated_weights, encrypted_weights])

        final_weights = decrypt_weights(encrypted_aggregated_weights, context, shape_map)
        net_glob.load_state_dict(final_weights)

        net_glob.eval()
        acc_train, _ = test_fun(net_glob, dataset_train, args)
        update_text(text_widget, f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}')

def create_gui(args):
    root = tk.Tk()
    root.title('Federated Learning Simulation')

    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=font.Font(family="San Francisco", size=16))
    text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def run_learning_process():
        start_time = time.time()
        context = setup_he_context()
        sequential_process(args, text_area, ax1, ax2, fig, canvas, context)
        total_time = time.time() - start_time
        text_area.insert(tk.END, f"Total time for completion: {total_time / 60:.2f} minutes.")

    thread = threading.Thread(target=run_learning_process)
    thread.start()

    root.mainloop()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    create_gui(args)
