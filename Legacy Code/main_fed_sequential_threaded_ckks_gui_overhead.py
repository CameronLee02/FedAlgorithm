import copy
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import scrolledtext, font, ttk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tenseal as ts
import platform
import psutil  # For system information
import networkx as nx
from models.Nets import MLP, Mnistcnn, Cifar10cnn
from models.Update import LocalUpdate
from models.test import test_fun
from utils.dataset_limit import get_dataset, exp_details
from utils.options import args_parser

# CKKS Context Setup
def create_ckks_context():
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
    for name, weight in weights_dict.items():
        if torch.isnan(weight).any():
            update_text(f"NaN detected in {description}: {name}", text_widget)
            return True
    return False

def log_weight_stats(weights_dict, description, text_widget):
    for name, weight in weights_dict.items():
        weight_np = weight.cpu().numpy()
        update_text(f"{description} - {name}: mean={weight_np.mean():.4f}, max={weight_np.max():.4f}, min={weight_np.min():.4f}", text_widget)

def encrypt_weights(weights_dict, context, text_widget, chunk_size=1000):
    start_time = time.time()
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
    
    encryption_time = time.time() - start_time
    update_text(f"Encryption completed in {encryption_time:.4f} seconds.", text_widget)
    overhead_info["encryption_times"].append(encryption_time)
    return encrypted_weights

def decrypt_weights(encrypted_weights, context, original_shapes, text_widget, client_count):
    start_time = time.time()
    update_text("Decrypting aggregated weights using CKKS decryption...", text_widget)
    decrypted_weights = {}
    for name, enc_weight_chunks in encrypted_weights.items():
        decrypted_flat = []
        for enc_weight in enc_weight_chunks:
            decrypted_flat.extend(enc_weight.decrypt())

        decrypted_array = (np.array(decrypted_flat, dtype=np.float32) / client_count).reshape(original_shapes[name])
        decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)

        log_weight_stats({name: decrypted_weights[name]}, "Decrypted Weights", text_widget)

        if check_for_nan({name: decrypted_weights[name]}, "Decrypted Weights", text_widget):
            raise ValueError(f"NaN detected in decrypted weights: {name}")

    decryption_time = time.time() - start_time
    update_text(f"Decryption completed in {decryption_time:.4f} seconds.", text_widget)
    overhead_info["decryption_times"].append(decryption_time)
    return decrypted_weights

def aggregate_encrypted_weights(encrypted_weights1, encrypted_weights2, client_count, text_widget):
    start_time = time.time()
    update_text("Aggregating encrypted weights using homomorphic addition...", text_widget)
    aggregated_weights = {}
    for name in encrypted_weights1.keys():
        aggregated_chunks = []
        for enc_w1, enc_w2 in zip(encrypted_weights1[name], encrypted_weights2[name]):
            aggregated_chunk = enc_w1 + enc_w2
            aggregated_chunks.append(aggregated_chunk)
        aggregated_weights[name] = aggregated_chunks
    
    aggregation_time = time.time() - start_time
    update_text(f"Encrypted weights aggregation completed in {aggregation_time:.4f} seconds.", text_widget)
    overhead_info["aggregation_times"].append(aggregation_time)
    return aggregated_weights

def client_training(client_id, dataset_train, dict_party_user, net_glob, text_widget, context, G, visualisation_canvas, visualisation_ax, colours, pos, received_encrypted_weights=None):
    update_text(f'Starting training on client {client_id}', text_widget)

    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[client_id])

    # Measure model distribution (downloading the model to the client)
    start_model_distribution = time.time()
    net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
    model_distribution_time = time.time() - start_model_distribution
    overhead_info["model_distribution_times"].append(model_distribution_time)

    update_text(f"Model distributed to client {client_id} in {model_distribution_time:.4f} seconds.", text_widget)

    # Training the local model
    local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

    if check_for_nan(local_weights, f"Client {client_id} local weights after training", text_widget):
        raise ValueError(f"NaN detected in Client {client_id} local weights after training.")

    #used to update the route/progress visualisation
    clients_index = list(G.nodes()).index(client_id)
    colours[clients_index] = "orange"
    update_display_network(G, visualisation_canvas, visualisation_ax, colours, pos)

    update_text(f'Client {client_id} has completed training. Loss: {loss:.4f}', text_widget)
    log_weight_stats(local_weights, f"Client {client_id} local weights before encryption", text_widget)

    if check_for_nan(local_weights, f"Client {client_id} local weights before encryption", text_widget):
        raise ValueError(f"NaN detected in Client {client_id} local weights before encryption.")

    # Encryption of local weights
    encrypted_weights = encrypt_weights(local_weights, context, text_widget)

    # Aggregation with previously received encrypted weights (if applicable)
    if received_encrypted_weights is not None:
        encrypted_weights = aggregate_encrypted_weights(encrypted_weights, received_encrypted_weights, 2, text_widget)
        colours[clients_index] = "green"
        update_display_network(G, visualisation_canvas, visualisation_ax, colours, pos)

    return encrypted_weights, local_weights, loss

def threaded_client_training(client_id, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results, G, visualisation_canvas, visualisation_ax, colours, pos):
    encrypted_weights, local_weights, loss = client_training(
        client_id,
        dataset_train,
        dict_party_user,
        net_glob,
        text_widget,
        context,
        G, #this parameter and below are used for route visualisation except received_encrypted_weights
        visualisation_canvas,
        visualisation_ax,
        colours,
        pos,
        received_encrypted_weights
    )
    results[client_id] = (encrypted_weights, local_weights, loss)

# Overhead Information Collection
overhead_info = {
    "epoch_times": [],
    "total_time": 0,
    "model_distribution_times": [],
    "encryption_times": [],
    "aggregation_times": [],
    "decryption_times": [],
    "update_times": [],
    "dataset_info": {},
    "system_info": {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_size": psutil.virtual_memory().total // (1024 ** 3)  # in GB
    }
}

def display_network(clients, visualisation_canvas, visualisation_ax):
    edge = []
    colours = ["red"] * len(clients)
    for i in range(1,len(clients)):
        edge.append((clients[i-1], clients[i]))
    G = nx.Graph()
    G.add_nodes_from(clients)
    G.add_edges_from(edge)
    pos ={node: (i, 0) for i, node in enumerate(G.nodes())}
    nx.draw(G, pos, with_labels=True, node_size=800, node_color=colours, font_size=10, font_weight="bold", edge_color="gray", ax=visualisation_ax)
    visualisation_canvas.draw()
    return colours, pos, G

def update_display_network(G, visualisation_canvas, visualisation_ax, colours, pos):
    nx.draw(G, pos, with_labels=True, node_size=800, node_color=colours, font_size=10, font_weight="bold", edge_color="gray", ax=visualisation_ax)
    visualisation_canvas.draw()

def sequential_process(args, text_widget, ax1, ax2, fig, canvas, visualisation_canvas, visualisation_ax):
    def update_plots(epoch_losses, epoch_accuracies):
        ax1.clear()
        ax2.clear()

        ax1.plot(epoch_losses, label='Average Loss per Epoch', marker='o')
        ax1.set_title('Training Loss Over Epochs', fontsize=8)
        ax1.set_xlabel('Epoch', fontsize=6)
        ax1.set_ylabel('Loss', fontsize=6)
        ax1.legend(fontsize=8)

        ax2.plot(epoch_accuracies, label='Accuracy per Epoch', marker='o')
        ax2.set_title('Training Accuracy Over Epochs', fontsize=8)
        ax2.set_xlabel('Epoch', fontsize=6)
        ax2.set_ylabel('Accuracy', fontsize=6)
        ax2.legend(fontsize=8)

        ax1.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(axis='both', which='major', labelsize=8)

        canvas.draw()

    dataset_train, dataset_test, dict_party_user, _ = get_dataset(args)
    overhead_info["dataset_info"] = {
        "dataset": args.dataset,
        "train_size": len(dataset_train),
        "test_size": len(dataset_test)
    }

    if args.model == 'cnn':
        if args.dataset == 'MNIST':
            net_glob = Mnistcnn(args=args).to(args.device)
        elif args.dataset == 'CIFAR10':
            net_glob = Cifar10cnn(args=args).to(args.device)
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

    net_glob.train()

    epoch_losses = []
    epoch_accuracies = []

    start_total_time = time.time()
    
    for iter in range(args.epochs):
        epoch_start_time = time.time()
        update_text(f'+++ Epoch {iter + 1} starts +++', text_widget)
        idxs_users = list(range(args.num_users))
        np.random.shuffle(idxs_users)
        colours, pos, G = display_network(idxs_users, visualisation_canvas, visualisation_ax)

        local_losses = []
        original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

        received_encrypted_weights = None
        results = {}
        threads = []

        for idx, user in enumerate(idxs_users):
            thread = threading.Thread(
                target=threaded_client_training,
                args=(user, dataset_train, dict_party_user, net_glob, context, text_widget, received_encrypted_weights, results, G, visualisation_canvas, visualisation_ax, colours, pos)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        client_count = 0

        for idx, user in enumerate(idxs_users):
            encrypted_weights, local_weights, loss = results[user]
            client_count += 1

            if received_encrypted_weights is not None:
                received_encrypted_weights = aggregate_encrypted_weights(
                    received_encrypted_weights,
                    encrypted_weights,
                    client_count,
                    text_widget
                )
            else:
                received_encrypted_weights = encrypted_weights
            colours[idx] = "green"
            update_display_network(G, visualisation_canvas, visualisation_ax, colours, pos)

            local_losses.append(loss)

        update_text('Final client sending aggregated encrypted weights to server.', text_widget)
        decrypted_weights = decrypt_weights(received_encrypted_weights, context, original_shapes, text_widget, client_count)

        if check_for_nan(decrypted_weights, "Global Model Weights", text_widget):
            raise ValueError("NaN detected in global model weights before updating.")
        
        log_weight_stats(decrypted_weights, "Global Model Weights", text_widget)

        net_glob.load_state_dict(decrypted_weights)
        update_text('Server has updated the global model with final aggregated weights.', text_widget)

        net_glob.eval()
        acc_train, _ = test_fun(net_glob, dataset_train, args)
        epoch_losses.append(np.mean(local_losses))
        epoch_accuracies.append(acc_train)

        update_text(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
        update_plots(epoch_losses, epoch_accuracies)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        overhead_info["epoch_times"].append(epoch_duration)
        visualisation_ax.clear() #used to clear/reset the network visualisation window
        visualisation_canvas.draw()

    overhead_info["total_time"] = time.time() - start_total_time

    exp_details(args)
    update_text("Training complete. Final Accuracy: {:.2f}".format(acc_train), text_widget)
    show_results_window(overhead_info)

def show_results_window(info):
    result_window = tk.Toplevel()
    result_window.title("Overhead Evaluation Results")

    title_font = font.Font(family="San Francisco", size=18, weight="bold")
    content_font = font.Font(family="San Francisco", size=12)

    tk.Label(result_window, text="Overhead Evaluation Results", font=title_font).pack(pady=10)
    
    results_text = tk.Text(result_window, wrap=tk.WORD, font=content_font)
    results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    results_text.insert(tk.END, "System Information:\n")
    for key, value in info["system_info"].items():
        results_text.insert(tk.END, f"{key}: {value}\n")

    results_text.insert(tk.END, "\nDataset Information:\n")
    for key, value in info["dataset_info"].items():
        results_text.insert(tk.END, f"{key}: {value}\n")

    results_text.insert(tk.END, "\nOverhead Information:\n")
    results_text.insert(tk.END, f"Total Time: {info['total_time']:.2f} seconds\n")
    for i, epoch_time in enumerate(info["epoch_times"]):
        results_text.insert(tk.END, f"Epoch {i + 1} Time: {epoch_time:.2f} seconds\n")

    results_text.insert(tk.END, "\nModel Distribution Times:\n")
    for i, model_time in enumerate(info["model_distribution_times"]):
        results_text.insert(tk.END, f"Epoch {i + 1} Model Distribution Time: {model_time:.2f} seconds\n")

    results_text.insert(tk.END, "\nEncryption Times:\n")
    for i, enc_time in enumerate(info["encryption_times"]):
        results_text.insert(tk.END, f"Epoch {i + 1} Encryption Time: {enc_time:.2f} seconds\n")

    results_text.insert(tk.END, "\nAggregation Times:\n")
    for i, agg_time in enumerate(info["aggregation_times"]):
        results_text.insert(tk.END, f"Epoch {i + 1} Aggregation Time: {agg_time:.2f} seconds\n")

    results_text.insert(tk.END, "\nDecryption Times:\n")
    for i, dec_time in enumerate(info["decryption_times"]):
        results_text.insert(tk.END, f"Epoch {i + 1} Decryption Time: {dec_time:.2f} seconds\n")

    results_text.config(state=tk.DISABLED)

class GUI():
    def __init__(self, args):
        self.args = args

    def create_gui(self):
        '''Create the GUI window for the Federated Learning process'''
        root = tk.Tk()
        root.title('Federated Learning Simulation with CKKS Encryption')

        custom_font = font.Font(family="San Francisco", size=16)
        text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=10, font=custom_font)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.visualisation_window = tk.Toplevel(root)
        self.visualisation_window.minsize(800,600)
        self.visualisation_window.title("Route Visualisation Window")
        self.visualisation_window.protocol("WM_DELETE_WINDOW", lambda: None)
        self.visualisation_window.withdraw()

        frame = tk.Frame(self.visualisation_window)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        l1 = tk.Label(frame, text = "Red -> Currently Training", fg="red")
        l1.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        l2 = tk.Label(frame, text = "Orange -> Finished Training, Waiting for Aggregation", fg="orange")
        l2.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        l3 = tk.Label(frame, text = "Green -> Finished Aggregation", fg="green")
        l3.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        fig, visualisation_ax = plt.subplots(figsize=(6, 4))
        visualisation_ax.set_title("Route Visualization")
        visualisation_canvas = FigureCanvasTkAgg(fig, master=self.visualisation_window)
        visualisation_canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        def open_route_visualisation():
            if self.visualisation_window.state() == "withdrawn":
                self.visualisation_window.deiconify()
            else:
                print("Cannot open Visualisation Window. It is already open")

        frame = tk.Frame(root, bg='lightblue')
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        route_visualisation_btn = tk.Button(frame, 
            text ="Click to open Route Visualisation Window", 
            command = open_route_visualisation)
        
        route_visualisation_btn.pack(side=tk.BOTTOM, padx=10, pady=5)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        def run_learning_process():
            start_time = time.time()
            sequential_process(self.args, text_area, ax1, ax2, fig, canvas, visualisation_canvas, visualisation_ax)
            total_time = time.time() - start_time
            text_area.insert(tk.END, f"Total time for completion: {total_time / 60:.2f} minutes.")

        thread = threading.Thread(target=run_learning_process)
        thread.start()

        root.mainloop()

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    gui = GUI(args)
    gui.create_gui()

# End of File