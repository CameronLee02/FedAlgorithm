import threading
import time
import tenseal as ts
import numpy as np
import torch
import networkx as nx
from models.test import test_fun
import statistics
import csv
import os
import gc

class ServerNodeClass():
    def __init__(self, node_id, network, args):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = []
        self.route = []
        self.predecessors = []
        self.received_encrypted_weights_list = []
        self.local_loss = []
    
    #This function collects all the nodes that are in the network
    def getNodeList(self, node_list):
        self.node_list = node_list 
    
    #This function is used to as a way to receive messages from client nodes
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "VALIDATED_POW" in message.keys() and sender_id in self.node_list.keys():
            pow_results = message["VALIDATED_POW"]
            if pow_results == True:
                self.overhead_info["pow_time"].append(time.time() - self.pow_start_time)
                self.startRouteCalc(False)
        
        if "VALIDATED_NODE_ROUTE" in message.keys() and sender_id in self.node_list.keys():
            self.route = message["VALIDATED_NODE_ROUTE"]
            self.predecessors = [] #holds the IDs of all the central server's predecessors. So it knows who to look out for
            for route in self.route:
                self.predecessors.append(route[-1])
            self.overhead_info["route_generation_time"].append(time.time() - self.route_start_time)
        
        if "ENCRYPTED_WEIGHTS" in message.keys() and sender_id in self.node_list.keys() and sender_id in self.predecessors:
            self.received_encrypted_weights_list.append(message["ENCRYPTED_WEIGHTS"][0])
            self.local_loss += message["ENCRYPTED_WEIGHTS"][1]
        
        if "FINAL_NOISE_VALUE" in message.keys() and sender_id in self.node_list:
            self.noise_values_count += 1
            for index, route in enumerate(self.route):
                if sender_id in route:
                    self.noise_values[index].append(message["FINAL_NOISE_VALUE"])
    
    #This function is used to simulate the central server sending a list of the participating to all the nodes 
    def sendOutListOfNodes(self):
        if self.args.pow:
            print("Sent out the pow start message") 
            self.pow_start_time = time.time()
            self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())}, "pow")
        else:
            self.startRouteCalc(True)
        
    def startRouteCalc(self, send_list):
        print("Sent out the route start message")
        self.route_start_time = time.time()
        if send_list:
            self.network.messageAllNodesExcludeServer(0, {"START_ROUTE_GEN" : list(self.node_list.keys())}, "route")
        else:
            self.network.messageAllNodesExcludeServer(0, {"START_ROUTE_GEN" : None}, "route")
    
    # CKKS Context Setup
    def create_ckks_context(self):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        return context
    
    def decryptWeights(self, encrypted_weights, context, original_shapes, text_widget, client_count, noise):
        start_time = time.time()
        self.network.updateText("Decrypting aggregated weights using CKKS decryption...", text_widget)
        decrypted_weights = {}
        for name, enc_weight_chunks in encrypted_weights.items():
            decrypted_flat = []
            for enc_weight in enc_weight_chunks:
                decrypted_flat.extend(enc_weight.decrypt())

            decrypted_array = ((np.array(decrypted_flat, dtype=np.float32) / client_count) - noise / client_count).reshape(original_shapes[name]) ### REMOVE NOISE HERE
            decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)

            #self.network.logWeightStats({name: decrypted_weights[name]}, "Decrypted Weights", text_widget)

            if self.network.checkForNan({name: decrypted_weights[name]}, "Decrypted Weights", text_widget):
                raise ValueError(f"NaN detected in decrypted weights: {name}")

        decryption_time = time.time() - start_time
        self.network.updateText(f"Decryption completed in {decryption_time:.4f} seconds.", text_widget)
        self.overhead_info["decryption_times"].append(decryption_time)
        return decrypted_weights
    
    def displayNetwork(self, visualisation_canvas, visualisation_ax):

        nodes = []
        edges = []
        for partition in self.route:
            nodes += partition
            for node in range(1,len(partition)):
                edges.append((partition[node-1], partition[node]))
        
        colours = ["red"] * len(nodes)
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)


        # Draw the graph on the specified axes
        pos = {}
        for i in range(len(self.route)):
            for j in range(len(self.route[i])):
                pos[self.route[i][j]] = (j, -i)
        
        nx.draw(G, pos, with_labels=True, node_size=800, node_color=colours, font_size=10, font_weight="bold", edge_color="gray", ax=visualisation_ax)
        visualisation_canvas.draw()
        return colours, pos, G

    def calculateNoise(self):
        noise_start_time = time.time()
        self.noise_values = []
        self.noise_values_count = 0 #keeps track of how many nodes have sent back their calculated noise
        max_noise_count = 0 #contains the max amount of nodes that will send back their calculated noise

        for route in self.route:
            max_noise_count += len(route)
            self.noise_values.append([])
        self.network.messageAllNodesExcludeServer(0, {"CALC_NOISE" : None}, "noise")

        #waits until it has received all the node's noise paritions sums
        while self.noise_values_count != max_noise_count :
            time.sleep(0.01)
        
        print(f"Central server received: {self.noise_values}")
        self.noise_added = 0
        for noise in self.noise_values:
            self.noise_added += statistics.mode(noise)
        print(f"Central server received: {self.noise_added }")
        self.overhead_info["noise_calc_time"].append(time.time() - noise_start_time)

    def updateOverheadDict(self, epoch_num):
        self.overhead_info["epoch_num"] = epoch_num

        self.overhead_info["pow_num_transmissions"].append(0)
        self.overhead_info["route_generation_num_transmissions"].append(0)
        self.overhead_info["noise_calc_num_transmissions"].append(0)
        self.overhead_info["other_num_transmissions"].append(0)
    
    def trainingProcess(self, net_glob, dataset_train, dict_party_user, text_widget, visualisation_canvas, visualisation_ax, overhead_info, ax1, ax2, canvas):
        self.overhead_info = overhead_info

        net_glob.train()

        epoch_losses = []
        epoch_accuracies = []

        start_total_time = time.time()
        
        for iter in range(self.args.epochs):
            self.updateOverheadDict(iter)

            epoch_start_time = time.time()

            key_gen_time = time.time()
            context = self.create_ckks_context()
            self.overhead_info["key_generation_time"].append(time.time() - key_gen_time)

            self.network.updateText(f'+++ Epoch {iter + 1} starts +++', text_widget)

            self.sendOutListOfNodes()

            colours, pos, G = self.displayNetwork(visualisation_canvas, visualisation_ax)
            
            original_shapes = {name: weight.shape for name, weight in net_glob.state_dict().items()}

            self.received_encrypted_weights_list = []
            self.local_loss = []
            threads = []

            #Collects all nodes currently participating in the training. some nodes may be in the network (self.node_list) but aren't participating
            nodes = []
            for route in self.route:
                nodes += route

            train_time_list = []
            encryption_time_list = []
            aggregate_time_list = []

            for node_id in nodes:
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

            self.network.updateText('Final clients sending aggregated encrypted weights to server.', text_widget)

            received_encrypted_weights = self.received_encrypted_weights_list[0]
            for i in range(1, len(self.received_encrypted_weights_list)):
                aggregateStartTime = time.time()
                received_encrypted_weights = self.network.aggregateEncryptedWeights(received_encrypted_weights, self.received_encrypted_weights_list[i], text_widget)
                aggregate_time_list.append(time.time() - aggregateStartTime)

            self.overhead_info["encryption_times"].append(statistics.mean(encryption_time_list))
            self.overhead_info["aggregation_times"].append(statistics.mean(aggregate_time_list))
            self.overhead_info["training_times"].append(statistics.mean(train_time_list))
            
            self.calculateNoise()

            decrypted_weights = self.decryptWeights(received_encrypted_weights, context, original_shapes, text_widget, len(nodes), self.noise_added)

            if self.network.checkForNan(decrypted_weights, "Global Model Weights", text_widget):
                raise ValueError("NaN detected in global model weights before updating.")
            
            #self.network.logWeightStats(decrypted_weights, "Global Model Weights", text_widget)

            update_start_time = time.time()
            net_glob.load_state_dict(decrypted_weights)
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

            self.network.setRouteVolunteer(None)
            self.network.setPoWVolunteer(None)
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
            metrics = ["pow_time", "route_generation_time", "noise_calc_time", "training_times", "key_generation_time",
                        "encryption_times", "decryption_times", "aggregation_times", "update_times",
                        "epoch_times", "total_time"]
            data_rows = zip(*[self.overhead_info[metric] for metric in metrics])
            write.writerow(metrics)
            write.writerows(data_rows)

        #writes the transmissions of each portion of the experiment to a file
        transmissionfile = os.path.join(self.args.output_directory, self.args.output_directory + "_transmissions.csv")
        with open(transmissionfile, 'w', newline='') as file:
                write = csv.writer(file)
                metrics = ["pow_num_transmissions", "route_generation_num_transmissions", "noise_calc_num_transmissions", "other_num_transmissions"]
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