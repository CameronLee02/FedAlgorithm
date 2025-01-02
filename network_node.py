
import tkinter as tk
from tkinter import scrolledtext, font, ttk
import numpy as np
import time
import threading
import torch
import tenseal as ts
import platform
import psutil  # For system information
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.Nets import MLP, Mnistcnn, Cifar10cnn
from utils.dataset import get_dataset, exp_details

class NetworkSimulationClass():
    def __init__(self, args):
        self.args = args
        self.nodes = {}
        self.server_node = None
        self.route_volunteer = None
        self.route_volunteer_lock = threading.Lock()
        self.pow_volunteer = None
        self.pow_volunteer_lock = threading.Lock()
        self.overhead_info = {
            "epoch_num": 0,
            "epoch_times": [],
            "acc_score": [],
            "loss_score": [],
            "total_time": [],
            "pow_time": [],
            "route_generation_time": [],
            "noise_calc_time": [],
            "training_times": [],
            "pow_num_transmissions": [],
            "route_generation_num_transmissions": [],
            "noise_calc_num_transmissions": [],
            "other_num_transmissions": [],
            "key_generation_time": [],
            "encryption_times": [],
            "aggregation_times": [],
            "decryption_times": [],
            "update_times": [],
            "pow_memory": [],
            "route_memory": [],
            "noise_memory": [],
            "key_gen_memory": [],
            "encryption_memory": [],
            "aggregation_memory": [],
            "decryption_memory": [],
            "update_memory": [],
            "dataset_info": {},
            "system_info": {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(logical=True),
                "ram_size": psutil.virtual_memory().total // (1024 ** 3)  # in GB
            }
        }
    
    #this function is used to send 1 message to 1 node
    def messageSingleNode(self, sender_id, receiver_id ,message, reason):
        if sender_id in self.nodes.keys() and receiver_id in self.nodes.keys():
            self.checkReason(reason)
            self.nodes[receiver_id].receiveMessage(sender_id, message)
    
    #This function is used to send a message to just the central server node
    def messageCentralServer(self, sender_id, message, reason):
        self.checkReason(reason)
        self.server_node.receiveMessage(sender_id, message)
    
    #this function is used to send a message to all the node, except the central server node.
    def messageAllNodesExcludeServer(self, sender_id, message, reason):
        threads = []
        for key, value in self.nodes.items():
            if key != sender_id:
                self.checkReason(reason)
                t = threading.Thread(target=value.receiveMessage, args=(sender_id, message))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
    
    #checks the reason for the transmissions and increments the count the keeps track of the number of transmissions for that reason
    #purely just for collecting data. not necessary for implementation
    def checkReason(self, reason):
        if reason == "pow":
            self.overhead_info["pow_num_transmissions"][self.overhead_info["epoch_num"]] += 1
        elif reason == "other":
            self.overhead_info["other_num_transmissions"][self.overhead_info["epoch_num"]] += 1
        elif reason == "route":
            self.overhead_info["route_generation_num_transmissions"][self.overhead_info["epoch_num"]] += 1
        elif reason == "noise":
            self.overhead_info["noise_calc_num_transmissions"][self.overhead_info["epoch_num"]] += 1
        
    def get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss

    def updateText(self, message, text_widget):
        text_widget.insert(tk.END, message + '\n')
        text_widget.see(tk.END)
    
    def checkForNan(self, weights_dict, description, text_widget):
        for name, weight in weights_dict.items():
            if torch.isnan(weight).any():
                self.updateText(f"NaN detected in {description}: {name}", text_widget)
                return True
        return False

    def logWeightStats(self, weights_dict, description, text_widget):
        for name, weight in weights_dict.items():
            weight_np = weight.cpu().numpy()
            self.updateText(f"{description} - {name}: mean={weight_np.mean():.4f}, max={weight_np.max():.4f}, min={weight_np.min():.4f}", text_widget)
    
    def encryptWeights(self, weights_dict, context, text_widget, noise, chunk_size=1000):
        start_time = time.time()
        self.updateText("Encrypting weights using CKKS encryption...", text_widget)
        encrypted_weights = {}
        for name, weight in weights_dict.items():
            weight_np = weight.cpu().numpy().astype(np.float64)
            if np.isnan(weight_np).any():
                self.updateText(f"NaN detected in {name} before encryption", text_widget)
                continue

            noisey_weight_np = weight_np + noise ### ADD NOISE HERE
            
            weight_chunks = np.array_split(noisey_weight_np.flatten(), max(1, len(noisey_weight_np.flatten()) // chunk_size))
            encrypted_chunks = [ts.ckks_vector(context, chunk) for chunk in weight_chunks]
            encrypted_weights[name] = encrypted_chunks
        
        encryption_time = time.time() - start_time
        self.updateText(f"Encryption completed in {encryption_time:.4f} seconds.", text_widget)
        return encrypted_weights

    def aggregateEncryptedWeights(self, encrypted_weights1, encrypted_weights2, text_widget):
        start_time = time.time()
        self.updateText("Aggregating encrypted weights using homomorphic addition...", text_widget)
        aggregated_weights = {}
        for name in encrypted_weights1.keys():
            aggregated_chunks = []
            for enc_w1, enc_w2 in zip(encrypted_weights1[name], encrypted_weights2[name]):
                aggregated_chunk = enc_w1 + enc_w2
                aggregated_chunks.append(aggregated_chunk)
            aggregated_weights[name] = aggregated_chunks
        
        aggregation_time = time.time() - start_time
        self.updateText(f"Encrypted weights aggregation completed in {aggregation_time:.4f} seconds.", text_widget)
        return aggregated_weights

    def updateDisplayNetwork(self, G, visualisation_canvas, visualisation_ax, colours, pos):
        nx.draw(G, pos, with_labels=True, node_size=800, node_color=colours, font_size=10, font_weight="bold", edge_color="gray", ax=visualisation_ax)
        visualisation_canvas.draw()
    
    def updatePlots(self, epoch_losses, epoch_accuracies, ax1, ax2, canvas):
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

    def initialiseLearningFixtures(self, text_widget, ax1, ax2, fig, canvas, visualisation_canvas, visualisation_ax):
        dataset_train, dataset_test, dict_party_user, _ = get_dataset(self.args)
        self.overhead_info["dataset_info"] = {
            "dataset": self.args.dataset,
            "train_size": len(dataset_train),
            "test_size": len(dataset_test)
        }
        print(self.overhead_info["dataset_info"])

        if self.args.model == 'cnn':
            if self.args.dataset == 'MNIST':
                net_glob = Mnistcnn(args=self.args).to(self.args.device)
            elif self.args.dataset == 'CIFAR10':
                net_glob = Cifar10cnn(args=self.args).to(self.args.device)
            else:
                self.updateText('Error: unrecognized dataset for CNN model', text_widget)
                return
        elif self.args.model == 'mlp':
            len_in = 1
            for dim in dataset_train[0][0].shape:
                len_in *= dim
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=self.args.num_classes).to(self.args.device)
        else:
            self.updateText('Error: unrecognized model', text_widget)
            return

        self.updateText('Federated Learning Simulation started. Initializing model architecture...\n', text_widget)
        self.updateText('Model architecture loaded and initialized. Starting training process on dataset: ' + self.args.dataset + '\n', text_widget)
        self.updatePlots([], [], ax1, ax2, canvas)

        acc_train = self.server_node.trainingProcess(net_glob, dataset_train, dict_party_user, text_widget, visualisation_canvas, visualisation_ax, self.overhead_info, ax1, ax2, canvas)

        exp_details(self.args)
        self.updateText("Training complete. Final Accuracy: {:.2f}".format(acc_train), text_widget)
        self.showResultsWindow()
    
    def showResultsWindow(self):
        info = self.overhead_info
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

        results_text.config(state=tk.DISABLED)

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

        def openRouteVisualisation():
            if self.visualisation_window.state() == "withdrawn":
                self.visualisation_window.deiconify()
            else:
                print("Cannot open Visualisation Window. It is already open")

        frame = tk.Frame(root, bg='lightblue')
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        route_visualisation_btn = tk.Button(frame, 
            text ="Click to open Route Visualisation Window", 
            command = openRouteVisualisation)
        
        route_visualisation_btn.pack(side=tk.BOTTOM, padx=10, pady=5)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


        def startLearningProcess():
            self.initialiseLearningFixtures(text_area, ax1, ax2, fig, canvas, visualisation_canvas, visualisation_ax)

        thread = threading.Thread(target=startLearningProcess)
        thread.start()

        root.mainloop()
    
    #Adds a node to the network. ID:0 is reserved for the central server
    def addNode(self, node):
        if node.node_id == 0:
            self.server_node = node
        else:
            self.nodes[node.node_id] = node
    
    def getNodes(self):
        return self.nodes
    
    def getRouteVolunteer(self):
        return self.route_volunteer
    
    def getRouteVolunteerLock(self):
        return self.route_volunteer_lock
    
    def setRouteVolunteer(self, volunteer):
        self.route_volunteer = volunteer
    
    def getPoWVolunteer(self):
        return self.pow_volunteer
    
    def getPoWVolunteerLock(self):
        return self.pow_volunteer_lock
    
    def setPoWVolunteer(self, volunteer):
        self.pow_volunteer = volunteer