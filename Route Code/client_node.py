import threading
import time
import random
import hashlib 


#Client node class that acts as the local clients participating in FL
class ClientNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.node_list = None
        self.route_volunteer = None
        self.route_volunteer_available = True

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
        if "ROUTE_PARAMETERS" in message.keys() and sender_id in self.node_list:
            self.route_volunteer_available = False
            self.route_volunteer = sender_id
            self.algorithm = message["ROUTE_PARAMETERS"]
            print(f"Node {self.node_id} has received the algorithm {self.algorithm}")
            self.route_salt = int(random.uniform(1000000000, 10000000000)) # generate a random value to use
            self.route_hash = self.calculateHash(self.route_salt, None, self.algorithm)

            print(f"{self.node_id} finshed their hash: {self.route_hash}")
            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_RESULTS": self.route_hash})
        
        #ROUTE_RESULTS message contains the results from each nodes route hash operation. This is collected by the Route Volunteer
        if "ROUTE_RESULTS" in message.keys() and sender_id in self.node_list:
            self.route_results[sender_id] = message["ROUTE_RESULTS"]
        
        #ORDERED_ROUTE_RESULTS message contains the complete ordered list of each nodes route hash value. It is sent from the route volunteer to every normal node.
        if "ORDERED_ROUTE_RESULTS" in message.keys() and sender_id in self.node_list and sender_id == self.route_volunteer:
            ordered_dict = message["ORDERED_ROUTE_RESULTS"]
            self.happy_about_route_results = True

            items = list(ordered_dict.items())
    
            # Get index of the node_id in the dictionary
            index = next((i for i, (key, _) in enumerate(items) if key == self.node_id), None)
            
            # Retrieve predecessor and successor's items
            self.predecessor_route_results = items[index - 1] if index > 0 else None
            self.successor_route_results = items[index + 1] if index < len(items) - 1 else None

            #This section is used to allow the node to check if the predecessor, successor, or route volnteer lied about any of the hash values
            #check their own recorded hash value
            if ordered_dict[self.node_id] != self.calculateHash(self.route_salt, None, self.algorithm):
                self.happy_about_route_results = False
                self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results})
                return
            threads = []
            if self.predecessor_route_results != None:
                receiver_id = self.predecessor_route_results[0]
                print(f"Node {self.node_id} Sending hash salt request to node {receiver_id}")
                t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, receiver_id, {"HASH_SALT_REQUETS": None}))
                t.start()
                threads.append(t)
            if self.successor_route_results != None:
                receiver_id = self.successor_route_results[0]
                print(f"Node {self.node_id} Sending hash salt request to node {receiver_id}")
                t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, receiver_id, {"HASH_SALT_REQUETS": None}))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()
            
            self.network.messageSingleNode(self.node_id, sender_id, {"ROUTE_HAPPINESS": self.happy_about_route_results})
            return
        
        #HASH_SALT_REQUETS message contains a request of the salt this node added to their route hash value so the sender can validate the hash
        if "HASH_SALT_REQUETS" in message.keys() and sender_id in self.node_list:
            self.network.messageSingleNode(self.node_id, sender_id, {"HASH_SALT_RESULT": self.route_salt})
        
        #HASH_SALT_RESULT contains the salt used by either the predecessor or the successor. This section determines if they match or not
        if "HASH_SALT_RESULT" in message.keys() and sender_id in self.node_list:
            hash = self.calculateHash(message["HASH_SALT_RESULT"], None, self.algorithm)
            if self.predecessor_route_results is not None:
                if sender_id == self.predecessor_route_results[0]:
                    if self.predecessor_route_results[1] != hash:
                        self.happy_about_route_results = False
            elif self.successor_route_results is not None:
                if sender_id == self.successor_route_results[0]:
                    if self.successor_route_results[1] != hash:
                        self.happy_about_route_results = False

        #ROUTE_HAPPINESS message contains info on if the nodes are happy with the route created. They have validated their predecessor's and successor's hash values
        if "ROUTE_HAPPINESS" in message.keys() and sender_id in self.node_list:
            if message["ROUTE_HAPPINESS"] != True: 
                self.nodes_are_happy_with_route = False
                    
    def beginRouteProcedure(self):
        #simulate the node waiting a random time
        #time.sleep(random.uniform(1, 5))

        #making 1 be volunteer every reound due to latency in 'transmissions', which causes multiple nodes to volunteer at the same time
        if self.route_volunteer_available == True and self.node_id == 1:
            print(f"Node {self.node_id} can volunteer to lead in-house route calculation")
            self.routeVolunteer()
            self.route_volunteer_available == False
            self.route_volunteer = self.node_id
    
    def routeVolunteer(self):
        algorithm = "sha256" #Can add more hashing algorithms to improve security. malicious nodes are less likely to have pre hashed values if diff type of algorithms can be used
        
        self.route_results = {}
        threads = []
        for node in self.node_list:
            print(f"Sending algorithm to node {node}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ROUTE_PARAMETERS": algorithm}))
            t.start()
            threads.append(t)
        
        #route volunteer must make their own hash as well as this defines the order of the chain
        self.route_salt = int(random.uniform(1000000000, 10000000000)) # generate a random value to use
        hash = self.calculateHash(self.route_salt, None, algorithm)
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
            print(f"Sending Sorted Hash values to Node {node}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ORDERED_ROUTE_RESULTS": sorted_items}))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        if self.nodes_are_happy_with_route:
            print("All Nodes are happy with the route")
            self.network.messageCentralServer(self.node_id, {"VALIDATED_NODE_ROUTE": list(sorted_items.keys())}) #Sends the route to the central server
        else:
            print("Some nodes aren't happy with the route")
    
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
    
    def checkStatus(self):
        print("still alive")