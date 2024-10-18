import threading
import time
import random
import hashlib 


class ServerNodeClass(threading.Thread):
    def __init__(self, node_id, network, malicous):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.malicous = malicous
        self.running = True
        self.real_node_list = None
        self.pow_volunteer = None
        self.route_volunteer = None
        self.pow_volunteer_available = True
        self.route_volunteer_available = True
        self.workToBeDone = []
    
    def run(self):
        while self.running:
            if self.workToBeDone != []:
                work = self.workToBeDone[0]
                if "PoW" in work.keys():
                    print("Started PoW for", work["RECEIVER_ID"])

                    self.proofOfWork(work['PoW'], work["RECEIVER_ID"], work["SENDER_ID"])

                    self.workToBeDone.pop(0)
                if "ROUTE" in work.keys():
                    print("Node", work["RECEIVER_ID"], "making hash")

                    self.routeValues[work["RECEIVER_ID"]] = int(random.uniform(1000000000, 10000000000)) # generate a random value to use and store in a dictionary containing other fake nodes's random values
                    hash = self.calculateHash(self.routeValues[work["RECEIVER_ID"]], None, work["ROUTE"])
                    self.network.messageSingleNode(work["RECEIVER_ID"], work["SENDER_ID"], {"ROUTE_RESULTS": hash})

                    self.workToBeDone.pop(0)
    
    def stop(self):
        self.running = False
        
    def getNodeList(self):
        self.real_node_list = list(self.network.nodes.keys())
        self.fake_node_list = []#[self.real_node_list[-1] + 1, self.real_node_list[-1] + 2] ## Temp implementation. creates 2 fake nodes
    
    def receiveMessage(self, sender_id, message, receiver_id):
        if len(message.keys()) != 1:
            return

        if "NODE_LIST" in message.keys() and sender_id == 0:
            #Server doesn't need to check if node_list is valid a it is acting maliciously by pretending to be another node
            #Server still wants to go through validation stage so it has the chance to give out the PoW parameters
            test_type = message["NODE_LIST"]["TEST_TYPE"]
            self.validateNodesOnList(receiver_id, test_type)

        if "ROUTE_PARAMETERS" in message.keys() and sender_id in self.real_node_list + self.fake_node_list:
            self.route_volunteer_available = False
            self.route_volunteer = sender_id
            algorithm = message["ROUTE_PARAMETERS"]
            print(f"Node {receiver_id} (Malicous node controlled by Server) has received the algorithm {algorithm}")
            self.routeValues = {}
            self.workToBeDone.append({"ROUTE": algorithm, "RECEIVER_ID": receiver_id, "SENDER_ID": sender_id})
            while self.workToBeDone != []:
                time.sleep(1)

        if "PoW_PARAMETERS" in message.keys() and sender_id in self.real_node_list + self.fake_node_list:
            self.pow_volunteer_available = False
            self.pow_volunteer = sender_id
            parameters = message["PoW_PARAMETERS"]
            print(f"Node {receiver_id} (Malicous node controlled by Server) has received the parameters {parameters}")
            self.workToBeDone.append({"PoW": parameters, "RECEIVER_ID": receiver_id, "SENDER_ID": sender_id})
            while self.workToBeDone != []:
                time.sleep(1)
        
        #PoW_RESULTS message contains the results from the PoW of a particular node. Contains the dict {HASH, NONCE}
        if "PoW_RESULTS" in message.keys() and sender_id in self.real_node_list + self.fake_node_list and sender_id in self.pow_parameters.keys():
            self.pow_parameters[sender_id].update(message["PoW_RESULTS"])
        
        if "ROUTE_RESULTS" in message.keys() and sender_id in self.node_list:
            self.route_results[sender_id] = message["ROUTE_RESULTS"]
    
    def validateNodesOnList(self, receiver_id, test_type):
        #simulate the node waiting a random time
        time.sleep(random.uniform(4, 5))
        if test_type == "pow" and self.pow_volunteer_available == True and self.node_id == 1:
            print(f"Node {receiver_id} (Malicous node controlled by Server) can volunteer to create PoW")
            self.proofOfWorkVolunteer(receiver_id)
            self.pow_volunteer_available == False
            self.pow_volunteer = receiver_id
        elif test_type == "route" and self.route_volunteer_available == True and self.node_id == 1:
            print(f"Node {receiver_id} can volunteer to lead in-house route calculation")
            self.routeVolunteer()
            self.route_volunteer_available == False
            self.route_volunteer = self.node_id
    
    def proofOfWork(self, parameters, receiver_id, sender_id):
        nonce = 0
        hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], nonce, parameters["ALGORITHM"])
        while hash[0:parameters["DIFFICULTY"]] != "0" * parameters["DIFFICULTY"]: 
            nonce += 1
            hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], nonce, parameters["ALGORITHM"])
        print(f"{receiver_id} finshed their proof of work: {hash}")
        self.network.messageSingleNode(receiver_id, sender_id, {"PoW_RESULTS": {"NONCE": nonce}})
    
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
        
    def proofOfWorkVolunteer(self, receiver_id):
        self.pow_parameters = {}
        for node in self.real_node_list:
            #paramters of PoW is [Random 10 digit number, number of leading 0's, hashing algorithm]
            self.pow_parameters[node] = {"ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), "DIFFICULTY": 4, "ALGORITHM" :'sha256'}
        
        #sends out the parameters to the innocent nodes via the network
        threads = []
        for key, value in self.pow_parameters.items():
            print(f"Recorded parameters {key}: {value}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(receiver_id, key, {"PoW_PARAMETERS": value}))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        #This part is used to check the PoW of the nodes to make sure they are correct and that they solved it in time
        for key, value in self.pow_parameters.items():
            #server doesn't need to actually check if the innocent nodes pass the PoW since it just wants to move onto next stage
            print(f"Node {key}'s PoW is correct and was completed in time")
    
    def routeVolunteer(self):
        algorithm = "sha256" #Can add more hashing algorithms to improve security. malicious nodes are less likely to have pre hashed values if diff type of algorithms can be used
        
        self.route_results = {}
        threads = []
        for node in self.real_node_list:
            print(f"Sending algorithm to node {node}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ROUTE_PARAMETERS": algorithm}))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        for key, value in self.route_results.items():
            print(f"Node {key} provided the hash value: {value}")

        print("--------------------------Ordering all hashing values in ascending order--------------------------")
        sorted_items = dict(sorted(self.route_results.items(), key=lambda item: item[1]))
        print(sorted_items)
        for key, value in sorted_items.items():
            print(f"Node {key} provided the hash value: {value}")
    
    #this function is used to simulate the central server sending a list of the participating to all the nodes 
    #Edit this function to allow the server to input dummy nodes into the list
    def sendOutListOfNodes(self, test_type):
        print("Sending out list of nodes")
        message = {"NODES": self.real_node_list + self.fake_node_list, "TEST_TYPE": test_type}
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : message})
    
    def getFakeListOfNodes(self):
        return self.fake_node_list