import threading
import time
import random
import hashlib 
import copy
import numpy as np

from models.Update import LocalUpdate

#Client node class that acts as the local clients participating in FL
class ClientNodeClass():
    def __init__(self, node_id, network, args):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.args = args
        self.node_list = None
        self.received_encrypted_weights = None
        self.route = None
        self.parition_numbers = []
        self.parition_sums = []
        self.noise_values = []


    #This function is used to as a way to receive messages from other client nodes or the central server
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "ENCRYPTED_WEIGHTS" in message.keys() and sender_id in self.node_list and sender_id == self.predecessor_id:
            self.received_encrypted_weights = message["ENCRYPTED_WEIGHTS"][0]
            self.local_losses = message["ENCRYPTED_WEIGHTS"][1]
        
        #CALC_NOISE message is a notification from the central server to tell the nodes to star the noise calculation process
        if "CALC_NOISE" in message.keys() and sender_id == 0:
            self.noiseCalculation()

        #NOISE_PARTITION message contains a node's share/parition of it's noise that they added to their results to protect it
        if "NOISE_PARTITION" in message.keys() and sender_id in self.node_list and sender_id in self.route:
            self.parition_numbers.append(message["NOISE_PARTITION"])

        #NOISE_PARTITION_SUM message contains a node's sum of the shares it received from other nodes
        if "NOISE_PARTITION_SUM" in message.keys() and sender_id in self.node_list and sender_id in self.route:
            self.parition_sums.append(message["NOISE_PARTITION_SUM"])

    def noise_generation(self):
        noise_partitions = []
        num_of_participates = len(self.route)
        for i in range(num_of_participates):
            noise_partitions.append(round(random.uniform(10,50),4))
        return noise_partitions
    
    #this function is used to split the noise added into partitions and send them to the other nodes to calculate the total noise added
    def noiseCalculation(self):
        print(f"Node {self.node_id} chose the noise: {self.noise} and split it into the values: {self.noise_partitions}")

        # sends all the noise number paritions (except 1) to all the other nodes in their route
        threads = []
        for index, node in enumerate(self.route):
            if self.node_id != node:
                t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"NOISE_PARTITION": self.noise_partitions[index]}, "noise"))
                t.start()
                threads.append(t)
            else:
                self.parition_numbers.append(self.noise_partitions[index])
        
        for t in threads:
            t.join()
        
        #waits until it has received all the other nodes noise paritions
        while len(self.parition_numbers) != len(self.route):
            time.sleep(0.01)
        
        parition_sum = sum(self.parition_numbers)
        self.parition_sums.append(parition_sum)

        # sends the sum of the paritions they have received to all the other nodes in their route
        threads = []
        for node in self.route:
            if self.node_id != node:
                t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"NOISE_PARTITION_SUM": parition_sum}, "noise"))
                t.start()
                threads.append(t)
        
        for t in threads:
            t.join()
        
        #waits until it has received all the other nodes noise parition sums
        while len(self.parition_sums) != len(self.route):
            time.sleep(0.01)
        
        final_noise_value = sum(self.parition_sums)

        #send final calculated noise to central server
        self.network.messageCentralServer(self.node_id, {"FINAL_NOISE_VALUE": final_noise_value}, "noise")

        #resets all object parameters used in this procedure
        self.parition_numbers = []
        self.parition_sums = []
        self.noise_values = []
        self.noise_partitions = []
        
    def client_training(self, route, client_id, dataset_train, dict_party_user, net_glob, text_widget, context, overhead_info, train_time_list, 
                        encryption_time_list, aggregate_time_list, weight_size_noise_list, 
                        G, visualisation_canvas, visualisation_ax, colours, pos):
        self.route = []
        self.node_list = []
        for partition in route:
            if self.node_id in partition:
                self.route = partition
            self.node_list += partition
        p = self.route.index(self.node_id)
        if p == 0:
            self.predecessor_id = None
        else: 
            self.predecessor_id = self.route[p-1]
        if p == len(self.route)-1:
            self.successor_id = None
        else: 
            self.successor_id = self.route[p+1]

        self.network.updateText(f'Starting training on client {client_id}', text_widget)
    
        local = LocalUpdate(args=self.args, dataset=dataset_train, idxs=dict_party_user[client_id])

        # Measure model distribution (downloading the model to the client)
        net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
        overhead_info["other_num_transmissions"][overhead_info["epoch_num"]] += 1 #count public key, route, and model distribution as one transmisison

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
        #self.network.logWeightStats(local_weights, f"Client {client_id} local weights before encryption", text_widget)

        if self.network.checkForNan(local_weights, f"Client {client_id} local weights before encryption", text_widget):
            raise ValueError(f"NaN detected in Client {client_id} local weights before encryption.")
        
        #generates noise that will be added to the encrypted weights
        self.noise_partitions = self.noise_generation()
        self.noise = round(sum(self.noise_partitions),4)

        #self.network.logWeightStats(local_weights, f"Node {self.node_id} Model Weights before encryption", text_widget)

        # Encryption of local weights
        startEncryptTime = time.time()
        encrypted_weights, weight_size_noise= self.network.encryptWeights(local_weights, context, text_widget, self.noise)
        encryption_time_list.append(time.time() - startEncryptTime)
        weight_size_noise_list.append(weight_size_noise)
        
        #This while loop is used to make the node wait for it's predecessor
        while self.received_encrypted_weights == None and self.predecessor_id != None:
            time.sleep(0.1)

        # Aggregation with previously received encrypted weights (if applicable)
        startAggregateTime = time.time()
        if self.predecessor_id is not None:
            current_encrypted_weights = self.network.aggregateEncryptedWeights(
                self.received_encrypted_weights,
                encrypted_weights,
                text_widget
            )
        else:
            current_encrypted_weights = encrypted_weights
            self.local_losses = []
        aggregate_time_list.append(time.time() - startAggregateTime)

        self.local_losses.append(loss)
        
        if self.successor_id is not None:
            self.network.messageSingleNode(self.node_id, self.successor_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]}, "other")
        else:
            self.network.messageCentralServer(self.node_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]}, "other")
        
        self.local_losses = [] #Resets the nodes recorded losses and encrypted weights for next epoch 
        self.received_encrypted_weights = None

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

