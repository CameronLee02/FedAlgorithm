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
            self.beginPoWProcedure()
        
        #PoW_PARAMETERS message contains the parameters for the PoW (the arbitrary number, num of leading 0's, hashing algorithm)
        #this section also performs the PoW 
        if "PoW_PARAMETERS" in message.keys() and sender_id in self.node_list:
            parameters = message["PoW_PARAMETERS"]

            hash, salt = self.proofOfWork(parameters)

            print(f"{self.node_id} finshed their proof of work: {hash}")
            self.network.messageSingleNode(self.node_id, sender_id, {"PoW_SINGLE_RESULT": {"SALT": salt}})

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
            self.network.messageSingleNode(self.node_id, sender_id, {"PoW_HAPPINESS": happy_about_pow_results})
        
        #PoW_HAPPINESS message contains info on if a node is happy with the pow results. They can do their own checking/validation
        if "PoW_HAPPINESS" in message.keys() and sender_id in self.node_list and sender_id in self.pow_parameters.keys():
            if message["PoW_HAPPINESS"] == False:
                self.nodes_are_happy_with_pow = False
        
        #PoW_HAPPINESS message contains info on if all nodes is happy with the pow results. So they can collectively go onto the next stage
        if "VALIDATED_POW" in message.keys() and sender_id in self.node_list:
            pass

    def beginPoWProcedure(self):
        #simulate the node waiting a random time
        time.sleep(random.uniform(0, 0.1))

        #making 1 be volunteer every reound due to latency in 'transmissions', which causes multiple nodes to volunteer at the same time
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
            self.pow_parameters[node] = {"NODE": node ,"ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), "DIFFICULTY": 5, "ALGORITHM" :'sha256'}

        #sends out the parameters to the nodes via the network
        threads = []
        self.times = {}
        for key, value in self.pow_parameters.items():
            print(f"Recorded parameters {key}: {value}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, key, {"PoW_PARAMETERS": value.copy()}))
            t.start()
            self.pow_parameters[key].update({"TIME_START": time.time()})
            threads.append(t)
        
        #PoW volunteer must conduct their own proof of work as well so they can be assigned a partitioned route later on
        self.pow_parameters[self.node_id] = {
            "ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), 
            "DIFFICULTY": 5, 
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
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"ORDERED_ROUTE_RESULTS": self.pow_parameters}))
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
            self.network.messageAllNodesExcludeServer(self.node_id, {"VALIDATED_POW": self.nodes_are_happy_with_pow})
        else:
            print("Some nodes aren't happy with the route")

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
    def calculateHash(self, parameter, salt, algorithm):
        if salt != None:
            if algorithm == 'sha256':
                sha = hashlib.sha256()
                input = f"{parameter}{salt}"
                sha.update(input.encode('utf-8'))
                return sha.hexdigest()
        else:
            if algorithm == 'sha256':
                sha = hashlib.sha256()
                sha.update(str(parameter).encode('utf-8'))
                return sha.hexdigest()