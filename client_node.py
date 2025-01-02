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
        
        #NODE_LIST message contains a list of all the participating nodes. node checks if its from the central server.
        #assume server node id = 0 for this program
        if "NODE_LIST" in message.keys() and sender_id == 0:
            self.node_list = message["NODE_LIST"].copy()

            #checks if it is in the list. if it isn't, ignore the message (invalid/untrusted list from server)
            if self.node_id not in self.node_list: 
                return
            #removes itself from list so it doesn't send messages to itself
            self.node_list.remove(self.node_id)
            self.beginPoWProcedure()
        
        #START_ROUTE_GEN message is a signal given to the clients to tell them to all start generating a route simultaneously 
        if "START_ROUTE_GEN" in message.keys() and sender_id == 0:
            if message["START_ROUTE_GEN"] != None:
                self.node_list = message["START_ROUTE_GEN"].copy()

                if self.node_id not in self.node_list: 
                    return

                self.node_list.remove(self.node_id)
            self.beginRouteProcedure()
        
        #ROUTE_PARAMETERS message is set from the route volunteer to all the other nodes and contains the information 
        #needed to generate a hash value for each in-house route operation
        if "ROUTE_PARAMETERS" in message.keys() and sender_id in self.node_list and sender_id == self.network.getRouteVolunteer():
            self.algorithm = message["ROUTE_PARAMETERS"]
            self.route_salt = int(random.uniform(1000000000, 10000000000)) # generate a random value to use
            self.route_hash = self.calculateHash(self.route_salt, None, self.algorithm)

            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_RESULTS": self.route_hash}, "route")
        
        #ROUTE_RESULTS message contains the results from each nodes route hash operation. This is collected by the Route Volunteer
        if "ROUTE_RESULTS" in message.keys() and sender_id in self.node_list:
            self.route_results[sender_id] = message["ROUTE_RESULTS"]
        
        #ORDERED_ROUTE_RESULTS message contains the complete ordered list of each nodes route hash value. It is sent from the route volunteer to every normal node.
        if "ORDERED_ROUTE_RESULTS" in message.keys() and sender_id in self.node_list and sender_id == self.network.getRouteVolunteer():
            ordered_dict = message["ORDERED_ROUTE_RESULTS"]["HASH_RESULTS"]
            partitions = message["ORDERED_ROUTE_RESULTS"]["PARTITIONS"]
            
            for partition in partitions:
                if self.node_id in partition:
                    self.route = partition
                    continue
            
            self.happy_about_route_results = True

            #This section is used to allow the node to check if the predecessor, successor, or route volnteer lied about any of the hash values
            #check their own recorded hash value
            if ordered_dict[self.node_id] != self.calculateHash(self.route_salt, None, self.algorithm):
                self.happy_about_route_results = False
                self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results}, "route")
                return

            self.validateNeighbours(ordered_dict)
            
            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results}, "route")
            return
        
        #ROUTE_HAPPINESS message contains info on if the nodes are happy with the route created. They have validated their predecessor's and successor's hash values
        if "ROUTE_HAPPINESS" in message.keys() and sender_id in self.node_list:
            if message["ROUTE_HAPPINESS"] != True: 
                self.nodes_are_happy_with_route = False

        #HASH_SALT_REQUETS message contains a request of the salt this node added to their route hash value so the sender can validate the hash
        if "HASH_SALT_REQUETS" in message.keys() and sender_id in self.node_list:
            self.network.messageSingleNode(self.node_id, sender_id, {"HASH_SALT_RESULT": self.route_salt}, "route")
        
        #HASH_SALT_RESULT contains the salt used by either the predecessor or the successor. This section determines if they match or not
        if "HASH_SALT_RESULT" in message.keys() and sender_id in self.node_list:
            hash = self.calculateHash(message["HASH_SALT_RESULT"], None, self.algorithm)
            if self.predecessor_route_results is not None:
                if sender_id == self.predecessor_route_results[0] and self.predecessor_route_results[1] != hash:
                    self.happy_about_route_results = False

            elif self.successor_route_results is not None:
                if sender_id == self.successor_route_results[0] and self.successor_route_results[1] != hash:
                    self.happy_about_route_results = False
        
        if "ENCRYPTED_WEIGHTS" in message.keys() and sender_id in self.node_list and sender_id == self.predecessor_id:
            self.received_encrypted_weights = message["ENCRYPTED_WEIGHTS"][0]
            self.local_losses = message["ENCRYPTED_WEIGHTS"][1]
        
        #CALC_NOISE message is a notification from the central server to tell the nodes to star the noise calculation process
        if "CALC_NOISE" in message.keys() and sender_id == 0:
            self.noiseProcedure()

        #NOISE_PARTITION message contains a node's share/parition of it's noise that they added to their results to protect it
        if "NOISE_PARTITION" in message.keys() and sender_id in self.node_list and sender_id in self.route:
            self.parition_numbers.append(message["NOISE_PARTITION"])

        #NOISE_PARTITION_SUM message contains a node's sum of the shares it received from other nodes
        if "NOISE_PARTITION_SUM" in message.keys() and sender_id in self.node_list and sender_id in self.route:
            self.parition_sums.append(message["NOISE_PARTITION_SUM"])
        
        #PoW_PARAMETERS message contains the parameters for the PoW (the arbitrary number, num of leading 0's, hashing algorithm)
        #this section also performs the PoW 
        if "PoW_PARAMETERS" in message.keys() and sender_id in self.node_list:
            parameters = message["PoW_PARAMETERS"]

            hash, salt = self.proofOfWork(parameters)

            print(f"{self.node_id} finshed their proof of work: {hash}")
            self.network.messageSingleNode(self.node_id, sender_id, {"PoW_SINGLE_RESULT": {"SALT": salt}}, "pow")

        #PoW_SINGLE_RESULT message contains the results (salt used) from the PoW of a single node
        if "PoW_SINGLE_RESULT" in message.keys() and sender_id in self.node_list and sender_id in self.pow_parameters.keys():
            self.pow_parameters[sender_id].update({"TIME_END": time.time()})
            self.pow_parameters[sender_id].update(message["PoW_SINGLE_RESULT"])

        #PoW_RESULTS message contains the results of all the participating nodes (Time, hash, provided number, etc)
        if "PoW_RESULTS" in message.keys() and sender_id in self.node_list and sender_id == self.network.getPoWVolunteer():
            results = message["PoW_RESULTS"]
            happy_about_pow_results = True
            ######
            #Here the node can do their own checking of the results and if they find anyone susupicious (eg. client spoofing as a normal node)
            ######
            if results == None:
                happy_about_pow_results = False
            self.network.messageSingleNode(self.node_id, sender_id, {"PoW_HAPPINESS": happy_about_pow_results}, "pow")
        
        #PoW_HAPPINESS message contains info on if a node is happy with the pow results. They can do their own checking/validation
        if "PoW_HAPPINESS" in message.keys() and sender_id in self.node_list and sender_id in self.pow_parameters.keys():
            if message["PoW_HAPPINESS"] == False:
                self.nodes_are_happy_with_pow = False
        
        #PoW_HAPPINESS message contains info on if all nodes is happy with the pow results. So they can collectively go onto the next stage
        if "VALIDATED_POW" in message.keys() and sender_id in self.node_list:
            pass
    
    #this function is used to split the noise added into partitions and send them to the other nodes to calculate the total noise added
    def noiseProcedure(self):
        num_of_participates = len(self.route)-1

        # I find this method of creating random partitions of a number is better at creating a more consistent/even spread. As others would have a 1 or 2 
        # very large valued partitions causing the rest to be very small (around the single digits). In a real world implementation, nodes can chose which every method they want
        list_of_partition_indexes = [] #holds the indexes that the noise number will be divided on. This creates numerous sub-lists where the size of these will be used to partition the noise number
        for i in range(num_of_participates):
            num = round(random.uniform(0, self.noise),4)
            while num in list_of_partition_indexes:
                num = round(random.uniform(0, self.noise),4)
            list_of_partition_indexes.append(num)
        list_of_partition_indexes.sort()
        partition_values = []
        for i in range(len(list_of_partition_indexes)):
            if i == 0:
                partition_values.append(list_of_partition_indexes[i])
            else:
                partition_values.append(round(list_of_partition_indexes[i] - list_of_partition_indexes[i-1],4))
        partition_values.append(round(self.noise - list_of_partition_indexes[-1],4))
        print(f"Node {self.node_id} chose the noise: {self.noise} and split it into the values: {partition_values}")

        # sends all the noise number paritions (except 1) to all the other nodes in their route
        threads = []
        for index, node in enumerate(self.route):
            if self.node_id != node:
                t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"NOISE_PARTITION": partition_values[index]}, "noise"))
                t.start()
                threads.append(t)
            else:
                self.parition_numbers.append(partition_values[index])
        
        for t in threads:
            t.join()
        
        #waits until it has received all the other nodes noise paritions
        while len(self.parition_numbers) != num_of_participates+1:
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
        while len(self.parition_sums) != num_of_participates+1:
            time.sleep(0.01)
        
        final_noise_value = sum(self.parition_sums)

        #send final calculated noise to central server
        self.network.messageCentralServer(self.node_id, {"FINAL_NOISE_VALUE": final_noise_value}, "noise")

        #resets all object parameters used in this procedure
        self.parition_numbers = []
        self.parition_sums = []
        self.noise_values = []

                
    #This function is used to check the hash values that the node's neighbours provided
    def validateNeighbours(self, ordered_dict):
        # Get index of the node_id in the dictionary
        self.position_in_the_route = self.route.index(self.node_id)

        self.predecessor_id = self.route[self.position_in_the_route - 1] if self.position_in_the_route > 0 else None
        self.successor_id = self.route[self.position_in_the_route + 1] if self.position_in_the_route < len(self.route) - 1 else None

        threads = []
        self.predecessor_route_results = None
        self.successor_route_results = None
        if self.predecessor_id != None:
            self.predecessor_route_results = ordered_dict[self.predecessor_id]
            print(f"Node {self.node_id} Sending hash salt request to node {self.predecessor_id}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, self.predecessor_id, {"HASH_SALT_REQUETS": None}, "route"))
            t.start()
            threads.append(t)
        if self.successor_id != None:
            self.successor_route_results = ordered_dict[self.successor_id] 
            print(f"Node {self.node_id} Sending hash salt request to node {self.successor_id}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, self.successor_id, {"HASH_SALT_REQUETS": None}, "route"))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
                    
    def beginRouteProcedure(self):      
        #simulate the node waiting a random time
        time.sleep(random.uniform(0, 0.1))

        if self.network.getRouteVolunteer() == None:
            with self.network.getRouteVolunteerLock():
                if self.network.getRouteVolunteer() == None: #re-check if no other node has volunteered yet
                    print(f"Node {self.node_id} can volunteer to lead in-house route calculation")
                    self.network.setRouteVolunteer(self.node_id)
                    self.routeVolunteer()
    
    def routeVolunteer(self):
        self.algorithm = "sha256" #Can add more hashing algorithms to improve security. malicious nodes are less likely to have pre hashed values if diff type of algorithms can be used
        
        self.route_results = {}
        threads = []
        for node in self.node_list:
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ROUTE_PARAMETERS": self.algorithm}, "route"))
            t.start()
            threads.append(t)
        
        #route volunteer must make their own hash as well as this defines the order of the chain
        self.route_salt = int(random.uniform(1000000000, 10000000000)) # generate a random value to use
        hash = self.calculateHash(self.route_salt, None, self.algorithm)
        self.route_results[self.node_id] = hash
        
        for t in threads:
            t.join()
        
        print("--------------------------Unordered hashing values--------------------------")
        for key, value in self.route_results.items():
            print(f"Node {key} provided the hash value: {value}")

        print("--------------------------Ordering all hashing values in ascending order--------------------------")
        sorted_hash_results_dict = dict(sorted(self.route_results.items(), key=lambda item: item[1]))
        sorted_hash_results_list = list(sorted_hash_results_dict.keys())
        for key, value in sorted_hash_results_dict.items():
            print(f"Node {key} provided the hash value: {value}")

        num_of_nodes_in_each_partition = self.args.partition_size
        partitions = self.generatePartitions(sorted_hash_results_list, num_of_nodes_in_each_partition)

        self.nodes_are_happy_with_route = True
        threads = []
        for node in self.node_list:
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ORDERED_ROUTE_RESULTS": {"HASH_RESULTS" : sorted_hash_results_dict, "PARTITIONS": partitions }}, "route"))
            t.start()
            threads.append(t)

        for partition in partitions:
            if self.node_id in partition:
                self.route = partition
                continue
            
        self.validateNeighbours(sorted_hash_results_dict)
        
        for t in threads:
            t.join()
        
        if self.nodes_are_happy_with_route:
            print("All Nodes are happy with the route")
            self.network.messageCentralServer(self.node_id, {"VALIDATED_NODE_ROUTE": partitions}, "route") #Sends the route to the central server
        else:
            print("Some nodes aren't happy with the route")
            #In a real world implementation, the whole route generation procedure will start again
            exit()

    def generatePartitions(self, sorted_hash_results_list, num_of_nodes_in_each_partitions):
        num_of_nodes = len(sorted_hash_results_list)
        num_of_partitions = num_of_nodes // num_of_nodes_in_each_partitions
        remainder = num_of_nodes % num_of_nodes_in_each_partitions
        partition_sizes = np.full(num_of_partitions, num_of_nodes_in_each_partitions)

        count = 0
        while remainder != 0:
            if count >= len(partition_sizes):
                count = 0
            partition_sizes[count] += 1
            remainder-= 1

        partitions = []
        count = 0
        for i in range(num_of_partitions):
            partitions.append(sorted_hash_results_list[count:count + int(partition_sizes[i])])
            count += int(partition_sizes[i])

        return partitions

    def beginPoWProcedure(self):
        #simulate the node waiting a random time
        time.sleep(random.uniform(0, 0.1))

        if self.network.getPoWVolunteer() == None:
            with self.network.getPoWVolunteerLock():
                if self.network.getPoWVolunteer() == None: #re-check if no other node has volunteered yet
                    print(f"Node {self.node_id} can volunteer to lead PoW calculation")
                    self.network.setPoWVolunteer(self.node_id)
                    self.PoWVolunteer()
    
    def PoWVolunteer(self):
        self.pow_parameters = {}
        for node in self.node_list:
            #paramters of PoW is [Random 10 digit number, number of leading 0's, hashing algorithm]
            self.pow_parameters[node] = {"NODE": node ,"ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), "DIFFICULTY": 4, "ALGORITHM" :'sha256'}

        #sends out the parameters to the nodes via the network
        threads = []
        self.times = {}
        for key, value in self.pow_parameters.items():
            print(f"Recorded parameters {key}: {value}")
            self.pow_parameters[key].update({"TIME_START": time.time()})
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, key, {"PoW_PARAMETERS": value.copy()}, "pow"))
            t.start()
            threads.append(t)
        
        #PoW volunteer must conduct their own proof of work as well so they can be assigned a partitioned route later on
        self.pow_parameters[self.node_id] = {
            "ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), 
            "DIFFICULTY": 4, 
            "ALGORITHM" :'sha256', 
            "TIME_START": time.time(), 
            "TIME_END": None}

        hash, salt = self.proofOfWork(self.pow_parameters[self.node_id])

        print(f"{self.node_id} finshed their proof of work: {hash}")

        self.pow_parameters[self.node_id]["TIME_END"] = time.time()
        self.pow_parameters[self.node_id].update({"SALT": salt})

        for t in threads:
            t.join()

        #This part is used to check the PoW of the nodes to make sure they are correct and that they solved it in time
        for key, value in self.pow_parameters.items():
            hash = self.calculateHash(value["ARBITRARY_NUMBER"], value["SALT"], value["ALGORITHM"])
            time_taken = self.pow_parameters[key]["TIME_END"] - self.pow_parameters[key]["TIME_START"]
            if hash[0:value["DIFFICULTY"]] != "0" * value["DIFFICULTY"]:
                print(f"Node {key} provided incorrect with the time of: {time_taken}")
            else:
                print(f"Node {key}'s PoW is correct with the time of: {time_taken}")
        
        self.nodes_are_happy_with_pow = True
        threads = []
        for node in self.node_list:
            print(f"Sending PoW results to Node {node}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"PoW_RESULTS": self.pow_parameters}, "pow"))
            t.start()
            threads.append(t)
        
        ######
        #Here the PoW volunteer can do their own checking of the results and if they find anyone susupicious (eg. client spoofing as a normal node)
        ######        
        
        for t in threads:
            t.join()
        
        if self.nodes_are_happy_with_pow:
            print("All Nodes are happy with the PoW")
            #Sends the overall happiness of the PoW to every node. So they can collectively go onto the next stage
            self.network.messageCentralServer(self.node_id, {"VALIDATED_POW": self.nodes_are_happy_with_pow}, "pow")
        else:
            print("Some nodes aren't happy with the route")
            exit
    
    #PoW is completed by getting the node to concat the provided number with a number of their own into string format.
    #This is then hashed and if it doesn't have the required number of leading 0's (difficulty). 
    #The node must redo this process with a different number of their own
    def proofOfWork(self, parameters):
        salt = 0
        hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], salt, parameters["ALGORITHM"])
        while hash[0:parameters["DIFFICULTY"]] != "0" * parameters["DIFFICULTY"]: 
            salt += 1
            hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], salt, parameters["ALGORITHM"])

        return hash, salt
    
    #can add different hash algorithms later if needed
    def calculateHash(self, parameter, nonce, algorithm):
        if nonce != None:
            if algorithm == 'sha256':
                sha = hashlib.sha256()
                sha.update(str(parameter).encode('utf-8') + str(nonce).encode('utf-8'))
                return sha.hexdigest()
        else:
            if algorithm == 'sha256':
                sha = hashlib.sha256()
                sha.update(str(parameter).encode('utf-8'))
                return sha.hexdigest()
        
    def client_training(self, client_id, dataset_train, dict_party_user, net_glob, text_widget, context, overhead_info, train_time_list, encryption_time_list, aggregate_time_list, 
                        encryption_mem_list, aggregate_mem_list, G, visualisation_canvas, visualisation_ax, colours, pos):
        self.network.updateText(f'Starting training on client {client_id}', text_widget)
    
        local = LocalUpdate(args=self.args, dataset=dataset_train, idxs=dict_party_user[client_id])

        # Measure model distribution (downloading the model to the client)
        net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
        overhead_info["other_num_transmissions"][overhead_info["epoch_num"]] += 1 #count model distribution as apart of training section

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
        self.noise = round(random.uniform(10, 100),4)

        # Encryption of local weights
        encrypt_memory = self.network.get_memory_usage()
        startEncryptTime = time.time()
        encrypted_weights = self.network.encryptWeights(local_weights, context, text_widget, self.noise)
        encryption_time_list.append(time.time() - startEncryptTime)
        encryption_mem_list.append(self.network.get_memory_usage() - encrypt_memory)

        
        #This while loop is used to make the node wait for it's predecessor
        while self.received_encrypted_weights == None and self.predecessor_id != None:
            time.sleep(0.1)

        # Aggregation with previously received encrypted weights (if applicable)
        aggregate_memory = self.network.get_memory_usage()
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
        aggregate_mem_list.append(self.network.get_memory_usage() - aggregate_memory)

        self.local_losses.append(loss)
        
        if self.successor_id is not None:
            self.network.messageSingleNode(self.node_id, self.successor_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]}, "other")
        else:
            self.network.messageCentralServer(self.node_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]}, "other")
        
        self.local_losses = [] #Resets the nodes recorded losses and encrypted weights for next epoch 
        self.received_encrypted_weights = None

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

