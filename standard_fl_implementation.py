import torch
import tkinter as tk
from tkinter import scrolledtext, font, ttk
import numpy as np
import time
import threading
import tenseal as ts
import platform
import psutil  # For system information
import matplotlib.pyplot as plt
import networkx as nx
import statistics
import csv
import os
import copy

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from models.Nets import MLP, Mnistcnn, Cifar10cnn, SvhnCnn
from utils.dataset import get_dataset, exp_details
from utils.options import args_parser2
from models.Update import LocalUpdate
from models.test import test_fun
from models.Fed import FedAvg

'''
This file contains the code that simulates the basic/standard FL implementation
Only communication is between the CS and indiviudal nodes. No homomorphic encryption, series aggregation, PoW, Noise, route generation
This will be used to compare between our new proposed implementation and the basic/standard implementation FedAVG
'''

#Client node class that acts as the local clients participating in FL
class ClientNodeClass():
    def __init__(self, node_id, network, args):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = None
        
    def client_training(self, client_id, dataset_train, dict_party_user, net_glob, text_widget, context, overhead_info, train_time_list, 
                        encryption_time_list, aggregate_time_list, G, visualisation_canvas, visualisation_ax, colours, pos):
        self.network.updateText(f'Starting training on client {client_id}', text_widget)
    
        local = LocalUpdate(args=self.args, dataset=dataset_train, idxs=dict_party_user[client_id])

        # Measure model distribution (downloading the model to the client)
        net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
        overhead_info["num_transmissions"][overhead_info["epoch_num"]] += 1 #count model distribution as apart of training section

        # Training the local model
        startTrainTime = time.time()
        local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(self.args.device))
        train_time_list.append(time.time() - startTrainTime)

        if self.network.checkForNan(local_weights, f"Client {client_id} local weights after training", text_widget):
            raise ValueError(f"NaN detected in Client {client_id} local weights after training.")

        #used to update the route/progress visualisation
        clients_index = list(G.nodes()).index(client_id)
        colours[clients_index] = "orange"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

        self.network.updateText(f'Client {client_id} has completed training. Loss: {loss:.4f}', text_widget)

        if self.network.checkForNan(local_weights, f"Client {client_id} local weights before encryption", text_widget):
            raise ValueError(f"NaN detected in Client {client_id} local weights before encryption.")

        self.network.messageCentralServer(self.node_id, {"RESULTS": [local_weights, loss]}, "other")

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

class ServerNodeClass():
    def __init__(self, node_id, network, args):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = []
        self.received_weights_list = []
        self.local_loss = []
    
    #This function collects all the nodes that are in the network
    def getNodeList(self, node_list):
        self.node_list = node_list 
    
    #This function is used to as a way to receive messages from client nodes
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "RESULTS" in message.keys() and sender_id in self.node_list.keys():
            self.received_weights_list.append(message["RESULTS"][0])
            self.local_loss.append(message["RESULTS"][1])

    def meanAggregatedWeights(self, aggregated_weights, client_count, text_widget):
        start_time = time.time()
        self.network.updateText("Computing mean of aggregated weights...", text_widget)
        mean_weights = {}

        for name, weight in aggregated_weights.items():

            mean_weight = weight / client_count  
            mean_weights[name] = mean_weight.clone().detach().to(dtype=torch.float32)

            if self.network.checkForNan({name: mean_weights[name]}, "Mean Weights", text_widget):
                raise ValueError(f"NaN detected in mean weights: {name}")

        mean_time = time.time() - start_time
        self.network.updateText(f"Mean computation completed in {mean_time:.4f} seconds.", text_widget)
        return mean_weights
    
    def displayNetwork(self, visualisation_canvas, visualisation_ax):
        nodes = list(self.node_list.keys())
        colours = ["red"] * len(nodes)
        G = nx.Graph()
        G.add_nodes_from(nodes)

        # Draw the graph on the specified axes
        pos = {}
        for i in range(len(nodes)):
            pos[nodes[i]] = (i, 0)
        
        nx.draw(G, pos, with_labels=True, node_size=800, node_color=colours, font_size=10, font_weight="bold", edge_color="gray", ax=visualisation_ax)
        visualisation_canvas.draw()
        return colours, pos, G

    def updateOverheadDict(self, epoch_num):
        self.overhead_info["epoch_num"] = epoch_num
        self.overhead_info["num_transmissions"].append(0)
    
    def trainingProcess(self, net_glob, dataset_train, dict_party_user, text_widget, visualisation_canvas, visualisation_ax, overhead_info, ax1, ax2, canvas):
        self.overhead_info = overhead_info

        net_glob.train()

        epoch_losses = []
        epoch_accuracies = []

        start_total_time = time.time()
        
        for iter in range(self.args.epochs):
            self.updateOverheadDict(iter)

            epoch_start_time = time.time()

            context = None

            self.network.updateText(f'+++ Epoch {iter + 1} starts +++', text_widget)

            colours, pos, G = self.displayNetwork(visualisation_canvas, visualisation_ax)
            
            #original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

            self.received_weights_list = []
            self.local_loss = []
            threads = []

            train_time_list = []
            encryption_time_list = []
            aggregate_time_list = []

            for node_id in list(self.node_list.keys()):
                node_object = self.node_list[node_id]
                thread = threading.Thread(
                    target=node_object.client_training,
                    args=(node_id,
                        dataset_train,
                        dict_party_user,
                        net_glob,
                        text_widget,
                        context,
                        self.overhead_info,
                        train_time_list,
                        encryption_time_list,
                        aggregate_time_list,
                        G, #this parameter and below are used for route visualisation
                        visualisation_canvas,
                        visualisation_ax,
                        colours,
                        pos
                    )
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            self.network.updateText('Final clients sending aggregated weights to server.', text_widget)

            # update global weights
            aggregate_start_time = time.time()
            w_glob = FedAvg(self.received_weights_list)
            aggregate_end_time = time.time() - aggregate_start_time

            self.overhead_info["aggregation_times"].append(aggregate_end_time)
            self.overhead_info["training_times"].append(statistics.mean(train_time_list))

            update_start_time = time.time()
            net_glob.load_state_dict(w_glob)
            self.overhead_info["update_times"].append(time.time() - update_start_time)
            self.network.updateText('Server has updated the global model with final aggregated weights.', text_widget)

            net_glob.eval()
            acc_train, _ = test_fun(net_glob, dataset_train, self.args)
            epoch_losses.append(np.mean(self.local_loss))
            epoch_accuracies.append(acc_train)
            self.overhead_info["acc_score"].append(acc_train)
            self.overhead_info["loss_score"].append(np.mean(self.local_loss))

            self.network.updateText(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
            self.network.updatePlots(epoch_losses, epoch_accuracies, ax1, ax2, canvas)

            self.overhead_info["epoch_times"].append(time.time() - epoch_start_time)

            visualisation_ax.clear() #used to clear/reset the network visualisation window
            visualisation_canvas.draw()

        total_time_array = np.full(shape=self.overhead_info["epoch_num"]+1, fill_value=time.time() - start_total_time)
        self.overhead_info["total_time"] = total_time_array

        try:
            os.makedirs(self.args.output_directory)
        except FileExistsError:
            pass
            
        #writes the times of each portion of the experiment to a file
        timefile = os.path.join(self.args.output_directory, self.args.output_directory + "_times.csv")
        with open(timefile, 'w', newline='') as file:
            write = csv.writer(file)
            metrics = ["training_times", "aggregation_times", "update_times", "epoch_times", "total_time"]
            data_rows = zip(*[self.overhead_info[metric] for metric in metrics])
            write.writerow(metrics)
            write.writerows(data_rows)

        #writes the transmissions of each portion of the experiment to a file
        transmissionfile = os.path.join(self.args.output_directory, self.args.output_directory + "_transmissions.csv")
        with open(transmissionfile, 'w', newline='') as file:
                write = csv.writer(file)
                metrics = ["num_transmissions"]
                data_rows = zip(*[self.overhead_info[metric] for metric in metrics])
                write.writerow(metrics)
                write.writerows(data_rows)

        #writes the accuracy and loss of each epoch of the experiment to a file
        scoresfile = os.path.join(self.args.output_directory, self.args.output_directory + "_scores.csv")
        with open(scoresfile, 'w', newline='') as file:
                write = csv.writer(file)
                metrics = ["acc_score", "loss_score"]
                data_rows = zip(*[self.overhead_info[metric] for metric in metrics])
                write.writerow(metrics)
                write.writerows(data_rows)
        return acc_train

class NetworkSimulationClass():
    def __init__(self, args):
        self.args = args
        self.nodes = {}
        self.server_node = None
        self.overhead_info = {
            "epoch_num": 0,
            "epoch_times": [],
            "acc_score": [],
            "loss_score": [],
            "total_time": [],
            "training_times": [],
            "num_transmissions": [],
            "key_generation_time": [],
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
        if reason == "other":
            self.overhead_info["num_transmissions"][self.overhead_info["epoch_num"]] += 1

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

    def aggregateWeights(self, weights1, weights2, text_widget):
        start_time = time.time()
        self.updateText("Aggregating non-encrypted weights using standard addition...", text_widget)
        aggregated_weights = {}

        # Direct aggregation for non-encrypted weights without chunking
        for name in weights1.keys():
            aggregated_weight = weights1[name] + weights2[name]  # Assuming these are directly addable (e.g., NumPy arrays or tensors)
            aggregated_weights[name] = aggregated_weight

        aggregation_time = time.time() - start_time
        self.updateText(f"Non-encrypted weights aggregation completed in {aggregation_time:.4f} seconds.", text_widget)
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
            elif self.args.dataset == 'SVHN':
                net_glob = SvhnCnn(args=self.args).to(self.args.device)
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
        self.root.quit()
        exit()
    

    def create_gui(self):
        '''Create the GUI window for the Federated Learning process'''
        self.root = tk.Tk()
        self.root.title('Federated Learning Simulation with CKKS Encryption')

        custom_font = font.Font(family="San Francisco", size=16)
        text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=10, font=custom_font)
        text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.visualisation_window = tk.Toplevel(self.root)
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

        frame = tk.Frame(self.root, bg='lightblue')
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

        self.root.mainloop()
        exit()
    
    #Adds a node to the network. ID:0 is reserved for the central server
    def addNode(self, node):
        if node.node_id == 0:
            self.server_node = node
        else:
            self.nodes[node.node_id] = node
    
    def getNodes(self):
        return self.nodes 

if __name__=="__main__":
    args = args_parser2()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    network_simulation_instance = NetworkSimulationClass(args)

    #Adds central server to the network
    central_server = ServerNodeClass(0, network_simulation_instance, args)
    network_simulation_instance.addNode(central_server)

    #Add all the nodes to the network
    for num in range(args.num_users):
        new_node = ClientNodeClass(num+1, network_simulation_instance, args)
        network_simulation_instance.addNode(new_node)

    nodes = network_simulation_instance.getNodes()
    for node in nodes.keys():
        print('Node ID:', node, '... object:', nodes[node])
        
    central_server.getNodeList(nodes)
    network_simulation_instance.create_gui()