import threading
import time
import random
import hashlib 
import copy

from models.Update import LocalUpdate

#Client node class that acts as the local clients participating in FL
class ClientNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.node_list = None
        self.received_encrypted_weights = None


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
            self.beginRouteProcedure()
        
        #ROUTE_PARAMETERS message is set from the route volunteer to all the other nodes and contains the information 
        #needed to generate a hash value for each in-house route operation
        if "ROUTE_PARAMETERS" in message.keys() and sender_id in self.node_list and sender_id == self.network.getRouteVolunteer():
            self.algorithm = message["ROUTE_PARAMETERS"]
            self.route_salt = int(random.uniform(1000000000, 10000000000)) # generate a random value to use
            self.route_hash = self.calculateHash(self.route_salt, None, self.algorithm)

            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_RESULTS": self.route_hash})
        
        #ROUTE_RESULTS message contains the results from each nodes route hash operation. This is collected by the Route Volunteer
        if "ROUTE_RESULTS" in message.keys() and sender_id in self.node_list:
            self.route_results[sender_id] = message["ROUTE_RESULTS"]
        
        #ORDERED_ROUTE_RESULTS message contains the complete ordered list of each nodes route hash value. It is sent from the route volunteer to every normal node.
        if "ORDERED_ROUTE_RESULTS" in message.keys() and sender_id in self.node_list and sender_id == self.network.getRouteVolunteer():
            ordered_dict = message["ORDERED_ROUTE_RESULTS"]
            self.happy_about_route_results = True

            #This section is used to allow the node to check if the predecessor, successor, or route volnteer lied about any of the hash values
            #check their own recorded hash value
            if ordered_dict[self.node_id] != self.calculateHash(self.route_salt, None, self.algorithm):
                self.happy_about_route_results = False
                self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results})
                return

            self.validateNeighbours(ordered_dict)
            
            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results})
            return
        
        #HASH_SALT_REQUETS message contains a request of the salt this node added to their route hash value so the sender can validate the hash
        if "HASH_SALT_REQUETS" in message.keys() and sender_id in self.node_list:
            self.network.messageSingleNode(self.node_id, sender_id, {"HASH_SALT_RESULT": self.route_salt})
        
        #HASH_SALT_RESULT contains the salt used by either the predecessor or the successor. This section determines if they match or not
        if "HASH_SALT_RESULT" in message.keys() and sender_id in self.node_list:
            hash = self.calculateHash(message["HASH_SALT_RESULT"], None, self.algorithm)
            if self.predecessor_route_results is not None:
                if sender_id == self.predecessor_route_results[0] and self.predecessor_route_results[1] != hash:
                    self.happy_about_route_results = False

            elif self.successor_route_results is not None:
                if sender_id == self.successor_route_results[0] and self.successor_route_results[1] != hash:
                    self.happy_about_route_results = False

        #ROUTE_HAPPINESS message contains info on if the nodes are happy with the route created. They have validated their predecessor's and successor's hash values
        if "ROUTE_HAPPINESS" in message.keys() and sender_id in self.node_list:
            if message["ROUTE_HAPPINESS"] != True: 
                self.nodes_are_happy_with_route = False
        
        if "ENCRYPTED_WEIGHTS" in message.keys() and sender_id in self.node_list and sender_id == self.predecessor_id:
            self.received_encrypted_weights = message["ENCRYPTED_WEIGHTS"][0]
            self.local_losses = message["ENCRYPTED_WEIGHTS"][1]

                
    #This function is used to check the hash values that the node's neighbours provided
    def validateNeighbours(self, ordered_dict):
        items = list(ordered_dict.items())

        # Get index of the node_id in the dictionary
        self.position_in_the_route = next((i for i, (key, _) in enumerate(items) if key == self.node_id), None)

        # Retrieve predecessor and successor's items
        self.predecessor_route_results = items[self.position_in_the_route - 1] if self.position_in_the_route > 0 else None
        self.successor_route_results = items[self.position_in_the_route + 1] if self.position_in_the_route < len(items) - 1 else None

        threads = []
        self.predecessor_id = None #These are used to store the IDs of the predecessor and successor for later use so they know where to send/received encrypted training data
        self.successor_id = None
        if self.predecessor_route_results != None:
            self.predecessor_id = self.predecessor_route_results[0]
            print(f"Node {self.node_id} Sending hash salt request to node {self.predecessor_id}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, self.predecessor_id, {"HASH_SALT_REQUETS": None}))
            t.start()
            threads.append(t)
        if self.successor_route_results != None:
            self.successor_id = self.successor_route_results[0]
            print(f"Node {self.node_id} Sending hash salt request to node {self.successor_id}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, self.successor_id, {"HASH_SALT_REQUETS": None}))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()
                    
    def beginRouteProcedure(self):      
        #simulate the node waiting a random time
        time.sleep(random.uniform(0, 0.1))

        volunteer = self.network.getRouteVolunteer()
        if volunteer == None:
            with self.network.getRouteVolunteerLock(): #Thread locking used to remove race conditions
                volunteer = self.network.getRouteVolunteer()
                if volunteer == None: #re-check if no other node has volunteered yet
                    print(f"Node {self.node_id} can volunteer to lead in-house route calculation")
                    self.network.setRouteVolunteer(self.node_id)
                    self.routeVolunteer()
    
    def routeVolunteer(self):
        self.algorithm = "sha256" #Can add more hashing algorithms to improve security. malicious nodes are less likely to have pre hashed values if diff type of algorithms can be used
        
        self.route_results = {}
        threads = []
        for node in self.node_list:
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ROUTE_PARAMETERS": self.algorithm}))
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
        sorted_items = dict(sorted(self.route_results.items(), key=lambda item: item[1]))
        for key, value in sorted_items.items():
            print(f"Node {key} provided the hash value: {value}")

        self.nodes_are_happy_with_route = True
        threads = []
        for node in self.node_list:
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ORDERED_ROUTE_RESULTS": sorted_items}))
            t.start()
            threads.append(t)

        self.validateNeighbours(sorted_items)
        
        for t in threads:
            t.join()
        
        if self.nodes_are_happy_with_route:
            print("All Nodes are happy with the route")
            self.network.messageCentralServer(self.node_id, {"VALIDATED_NODE_ROUTE": list(sorted_items.keys())}) #Sends the route to the central server
        else:
            print("Some nodes aren't happy with the route")
            #In a real world implementation, the whole route generation procedure will start again
            exit()
    
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
        
    def client_training(self, client_id, dataset_train, dict_party_user, net_glob, text_widget, context, args, overhead_info, G, visualisation_canvas, visualisation_ax, colours, pos):
        self.network.updateText(f'Starting training on client {client_id}', text_widget)

        local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_party_user[client_id])

        # Measure model distribution (downloading the model to the client)
        start_model_distribution = time.time()
        net_glob.load_state_dict(copy.deepcopy(net_glob.state_dict()))
        model_distribution_time = time.time() - start_model_distribution
        overhead_info["model_distribution_times"].append(model_distribution_time)

        self.network.updateText(f"Model distributed to client {client_id} in {model_distribution_time:.4f} seconds.", text_widget)

        # Training the local model
        local_weights, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

        if self.network.checkForNan(local_weights, f"Client {client_id} local weights after training", text_widget):
            raise ValueError(f"NaN detected in Client {client_id} local weights after training.")

        #used to update the route/progress visualisation
        clients_index = list(G.nodes()).index(client_id)
        colours[clients_index] = "orange"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)

        self.network.updateText(f'Client {client_id} has completed training. Loss: {loss:.4f}', text_widget)
        self.network.logWeightStats(local_weights, f"Client {client_id} local weights before encryption", text_widget)

        if self.network.checkForNan(local_weights, f"Client {client_id} local weights before encryption", text_widget):
            raise ValueError(f"NaN detected in Client {client_id} local weights before encryption.")

        # Encryption of local weights
        encrypted_weights = self.network.encryptWeights(local_weights, context, text_widget)
        
        #This while loop is used to make the node wait for it's predecessor
        while self.received_encrypted_weights == None and self.predecessor_id != None:
            time.sleep(0.1)

        # Aggregation with previously received encrypted weights (if applicable)
        if self.predecessor_id is not None:
            current_encrypted_weights = self.network.aggregateEncryptedWeights(
                self.received_encrypted_weights,
                encrypted_weights,
                self.position_in_the_route+1,
                text_widget
            )
        else:
            current_encrypted_weights = encrypted_weights
            self.local_losses = []
        
        self.local_losses.append(loss)
        
        if self.successor_id is not None:
            self.network.messageSingleNode(self.node_id, self.successor_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]})
        else:
            self.network.messageCentralServer(self.node_id, {"ENCRYPTED_WEIGHTS": [current_encrypted_weights, self.local_losses.copy()]})
        
        self.local_losses = [] #Resets the nodes recorded losses and encrypted weights for next epoch 
        self.received_encrypted_weights = None

        colours[clients_index] = "green"
        self.network.updateDisplayNetwork(G, visualisation_canvas, visualisation_ax, colours, pos)
    
    def checkStatus(self):
        print("still alive")