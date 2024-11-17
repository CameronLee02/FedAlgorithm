import threading
import time
import tenseal as ts
import numpy as np
import torch
import networkx as nx
from models.test import test_fun

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
        
        if "VALIDATED_NODE_ROUTE" in message.keys() and sender_id in self.node_list.keys():
            self.route = message["VALIDATED_NODE_ROUTE"]
            self.predecessors = [] #holds the IDs of all the central server's predecessors. So it knows who to look out for
            for route in self.route:
                self.predecessors.append(route[-1])
        
        if "ENCRYPTED_WEIGHTS" in message.keys() and sender_id in self.node_list.keys() and sender_id in self.predecessors:
            self.received_encrypted_weights_list.append(message["ENCRYPTED_WEIGHTS"][0])
            self.local_loss.append(message["ENCRYPTED_WEIGHTS"][1])
    
    #This function is used to simulate the central server sending a list of the participating to all the nodes 
    def sendOutListOfNodes(self):
        print("Sent out the route message")
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
    
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
    
    def decryptWeights(self, encrypted_weights, context, original_shapes, text_widget, client_count, overhead_info):
        start_time = time.time()
        self.network.updateText("Decrypting aggregated weights using CKKS decryption...", text_widget)
        decrypted_weights = {}
        for name, enc_weight_chunks in encrypted_weights.items():
            decrypted_flat = []
            for enc_weight in enc_weight_chunks:
                decrypted_flat.extend(enc_weight.decrypt())

            decrypted_array = (np.array(decrypted_flat, dtype=np.float32) / client_count).reshape(original_shapes[name])
            decrypted_weights[name] = torch.tensor(decrypted_array, dtype=torch.float32)

            self.network.logWeightStats({name: decrypted_weights[name]}, "Decrypted Weights", text_widget)

            if self.network.checkForNan({name: decrypted_weights[name]}, "Decrypted Weights", text_widget):
                raise ValueError(f"NaN detected in decrypted weights: {name}")

        decryption_time = time.time() - start_time
        self.network.updateText(f"Decryption completed in {decryption_time:.4f} seconds.", text_widget)
        overhead_info["decryption_times"].append(decryption_time)
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
    
    def trainingProcess(self, net_glob, dataset_train, dict_party_user, text_widget, visualisation_canvas, visualisation_ax, overhead_info, ax1, ax2, canvas):
        context = self.create_ckks_context()

        net_glob.train()

        epoch_losses = []
        epoch_accuracies = []

        start_total_time = time.time()
        
        for iter in range(self.args.epochs):
            epoch_start_time = time.time()
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
                        overhead_info,
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

            received_encrypted_weights = self.received_encrypted_weights_list[0]
            for i in range(1, len(self.received_encrypted_weights_list)):
                received_encrypted_weights = self.network.aggregateEncryptedWeights(
                received_encrypted_weights,
                self.received_encrypted_weights_list[i],
                text_widget
            )

            self.network.updateText('Final client sending aggregated encrypted weights to server.', text_widget)
            decrypted_weights = self.decryptWeights(received_encrypted_weights, context, original_shapes, text_widget, len(nodes), overhead_info)

            if self.network.checkForNan(decrypted_weights, "Global Model Weights", text_widget):
                raise ValueError("NaN detected in global model weights before updating.")
            
            self.network.logWeightStats(decrypted_weights, "Global Model Weights", text_widget)

            net_glob.load_state_dict(decrypted_weights)
            self.network.updateText('Server has updated the global model with final aggregated weights.', text_widget)

            net_glob.eval()
            acc_train, _ = test_fun(net_glob, dataset_train, self.args)
            epoch_losses.append(np.mean(self.local_loss))
            epoch_accuracies.append(acc_train)

            self.network.updateText(f'Epoch {iter + 1} completed. Train Acc: {acc_train:.2f}', text_widget)
            self.network.updatePlots(epoch_losses, epoch_accuracies, ax1, ax2, canvas)

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            overhead_info["epoch_times"].append(epoch_duration)

            self.network.setRouteVolunteer(None)
            visualisation_ax.clear() #used to clear/reset the network visualisation window
            visualisation_canvas.draw()

        overhead_info["total_time"] = time.time() - start_total_time
        return acc_train