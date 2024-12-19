import tkinter as tk
from tkinter import Button
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import statistics

import time

class ServerNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.node_list = None
        self.noise_values = []
        
    def getNodeList(self, node_list):
        self.node_list = node_list
    
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "FINAL_NOISE_VALUE" in message.keys() and sender_id in self.node_list:
            self.noise_values.append(message["FINAL_NOISE_VALUE"])
    
    #this function is used to simulate the central server sending a list of the participating to all the nodes 
    #Edit this function to allow the server to input dummy nodes into the list
    def sendOutListOfNodes(self):
        print("Sending out list of nodes")
        start = time.time()
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
        print(f"Time taken : {time.time()-start}")

        #waits until it has received all the node's noise paritions sums
        while len(self.noise_values) != len(self.node_list):
            time.sleep(0.1)
        
        print(f"Central server received: {self.noise_values}")
        print(f"Central server received: {statistics.mode(self.noise_values)}")
    