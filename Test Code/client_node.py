import threading
import time
import random
import hashlib 


#Client node class that acts as the local clients participating in FL
class ClientNodeClass(threading.Thread):
    def __init__(self, node_id, network, malicous):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.malicous = malicous
        self.node_list = None
        self.pow_volunteer = None
        self.pow_volunteer_available = True

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
            self.validateNodesOnList()

        #PoW_VOLUNTEER message contains the volunteering node_id that will give a PoW to every other node
        if "PoW_VOLUNTEER" in message.keys() and sender_id in self.node_list:
            self.pow_volunteer_available = False
            self.pow_volunteer = sender_id
            print(f"Node {self.node_id} agrees")

        #PoW_PARAMETERS message contains the parameters for the PoW (the arbitrary number, num of leading 0's, hashing algorithm)
        #this section also performs the PoW 
        if "PoW_PARAMETERS" in message.keys() and sender_id in self.node_list and sender_id == self.pow_volunteer:
            parameters = message["PoW_PARAMETERS"]
            print(f"Node {self.node_id} has received the parameters {parameters}")
            print(f"Started PoW for {self.node_id}")

            hash, nonce = self.proofOfWork(parameters)

            print(f"{self.node_id} finshed their proof of work: {hash}")
            self.network.messageSingleNode(self.node_id, sender_id, {"PoW_RESULTS": {"NONCE": nonce}})
        
        #PoW_RESULTS message contains the results from the PoW of a particular node. Contains the dict {HASH, NONCE}
        if "PoW_RESULTS" in message.keys() and sender_id in self.node_list and sender_id in self.pow_parameters.keys() and self.node_id == self.pow_volunteer:
            self.times[sender_id]["TIME_END"] = time.time()
            self.pow_parameters[sender_id].update(message["PoW_RESULTS"])

    def proofOfWork(self, parameters):
        nonce = 0
        hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], nonce, parameters["ALGORITHM"])
        while hash[0:parameters["DIFFICULTY"]] != "0" * parameters["DIFFICULTY"]: 
            nonce += 1
            hash = self.calculateHash(parameters["ARBITRARY_NUMBER"], nonce, parameters["ALGORITHM"])

        return hash, nonce
                    
    def validateNodesOnList(self):
        #simulate the node waiting a random time
        time.sleep(random.uniform(1, 5))
        if self.pow_volunteer_available == True:
            print(f"Node {self.node_id} can volunteer to create PoW")
            self.network.messageAllNodesExcludeServer(self.node_id, {"PoW_VOLUNTEER": self.node_id})
            self.pow_volunteer_available == False
            self.pow_volunteer = self.node_id
            self.proofOfWorkVolunteer()
    
    def proofOfWorkVolunteer(self):
        self.pow_parameters = {}
        for node in self.node_list:
            #paramters of PoW is [Random 10 digit number, number of leading 0's, hashing algorithm]
            self.pow_parameters[node] = {"ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), "DIFFICULTY": 5, "ALGORITHM" :'sha256'}

        #sends out the parameters to the nodes via the network
        threads = []
        self.times = {}
        for key, value in self.pow_parameters.items():
            print(f"Recorded parameters {key}: {value}")
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, key, {"PoW_PARAMETERS": value}))
            t.start()
            self.times[key] = ({"TIME_START": time.time(), "TIME_END": None})
            threads.append(t)
        
        #PoW volunteer must conduct their own proof of work as well so they can be assigned a partitioned route later on
        self.pow_parameters[self.node_id] = {"ARBITRARY_NUMBER": int(random.uniform(1000000000, 10000000000)), "DIFFICULTY": 5, "ALGORITHM" :'sha256'}
        self.times[self.node_id] = ({"TIME_START": time.time(), "TIME_END": None})

        hash, nonce = self.proofOfWork(self.pow_parameters[self.node_id])

        print(f"{self.node_id} finshed their proof of work: {hash}")

        self.times[self.node_id]["TIME_END"] = time.time()
        self.pow_parameters[self.node_id].update({"NONCE": nonce})


        for t in threads:
            t.join()

        #This part is used to check the PoW of the nodes to make sure they are correct and that they solved it in time
        for key, value in self.pow_parameters.items():
            hash = self.calculateHash(value["ARBITRARY_NUMBER"], value["NONCE"], value["ALGORITHM"])
            time_taken = self.times[key]["TIME_END"] - self.times[key]["TIME_START"]
            if hash[0:value["DIFFICULTY"]] != "0" * value["DIFFICULTY"]:
                print(f"Node {key} provided incorrect calculation for the PoW or was too slow in calculating PoW with the time of: {time_taken}")
            else:
                print(f"Node {key}'s PoW is correct with the time of: {time_taken}")
    
    def calculateHash(self, parameter, nonce, algorithm):
        if algorithm == 'sha256':
            sha = hashlib.sha256()
            sha.update(str(parameter).encode('utf-8') + str(nonce).encode('utf-8'))
            return sha.hexdigest()
    
    def checkStatus(self):
        print("still alive")
