import threading
import time
import random
import hashlib 
import numpy as np


#Client node class that acts as the local clients participating in FL
class ClientNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.node_list = None
        self.parition_numbers = []
        self.parition_sums = []

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
            self.noiseProcedure()
        
        #NOISE_PARTITION message contains a node's share/parition of it's noise that they added to their results to protect it
        if "NOISE_PARTITION" in message.keys() and sender_id in self.node_list:
            self.parition_numbers.append(message["NOISE_PARTITION"])

        #NOISE_PARTITION_SUM message contains a node's sum of the shares it received from other nodes
        if "NOISE_PARTITION_SUM" in message.keys() and sender_id in self.node_list:
            self.parition_sums.append(message["NOISE_PARTITION_SUM"])
                    
    def noiseProcedure(self):
        self.noise = random.randrange(100, 1000)
        num_of_participates = len(self.node_list)#grabs the number of participates excluding themselves (NEED TO ADJUST SO ONLY COUNTS THE NUMBER IN ITS ROUTE)

        # I find this method of creating random partitions of a number is better at creating a more consistent/even spread. As others would have a 1 or 2 
        # very large valued partitions causing the rest to be very small (around the single digits). In a real world implementation, nodes can chose which every method they want
        list_of_partition_indexes = [] #holds the indexes that the noise number will be divided on. This creates numerous sub-lists where the size of these will be used to partition the noise number
        for i in range(num_of_participates):
            num = random.randrange(0, self.noise)
            while num in list_of_partition_indexes:
                num = random.randrange(0, self.noise)
            list_of_partition_indexes.append(num)
        list_of_partition_indexes.sort()
        partition_values = []
        for i in range(len(list_of_partition_indexes)):
            if i == 0:
                partition_values.append(list_of_partition_indexes[i])
            else:
                partition_values.append(list_of_partition_indexes[i] - list_of_partition_indexes[i-1])
        partition_values.append(self.noise - list_of_partition_indexes[-1])
        print(f"Node {self.node_id} chose the noise: {self.noise} and split it into the values: {partition_values}")

        # sends all the noise number paritions (except 1) to all their neighbours
        self.parition_numbers.append(partition_values[0])
        threads = []
        for index, node in enumerate(self.node_list):
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"NOISE_PARTITION": partition_values[index+1]}))
            t.start()
            threads.append(t)
        
        #waits until it has received all the other nodes noise paritions
        while len(self.parition_numbers) != num_of_participates+1:
            time.sleep(0.1)
        
        parition_sum = sum(self.parition_numbers)
        self.parition_sums.append(parition_sum)

        # sends the sum of the paritions they have received to all their neighbours
        threads = []
        for node in self.node_list:
            t = threading.Thread(target=self.network.messageSingleNode, args=(self.node_id, node, {"NOISE_PARTITION_SUM": parition_sum}))
            t.start()
            threads.append(t)
        
        #waits until it has received all the other nodes noise parition sums
        while len(self.parition_sums) != num_of_participates+1:
            time.sleep(0.1)
        
        final_noise_value = sum(self.parition_sums)

        #send final calculated noise to central server
        self.network.messageCentralServer(self.node_id, {"FINAL_NOISE_VALUE": final_noise_value})


    
