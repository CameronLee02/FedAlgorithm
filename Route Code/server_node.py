import tkinter as tk
from tkinter import Button
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import time

class ServerNodeClass():
    def __init__(self, node_id, network):
        super().__init__()
        self.node_id = node_id
        self.network = network
        self.running = True
        self.node_list = None

    def stop(self):
        self.running = False
        
    def getNodeList(self, node_list):
        self.node_list = node_list
    
    def receiveMessage(self, sender_id, message):
        if len(message.keys()) != 1:
            return
        
        if "VALIDATED_NODE_ROUTE" in message.keys() and sender_id in self.node_list:
            route = message["VALIDATED_NODE_ROUTE"]
            print(route)
            for key in self.node_list.keys():
                print(f"Node {key} has the predecessor: {self.node_list[key].predecessor_id} and the successor: {self.node_list[key].successor_id}")
            
            # Set up Tkinter window
            root = tk.Tk()
            root.title("Network Graph Display")

            # Create a matplotlib figure and axes
            fig, ax = plt.subplots(figsize=(5, 5))
            canvas = FigureCanvasTkAgg(fig, master=root)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Button to create and display the network graph
            create_button = Button(root, text="Create Network Graph", command=lambda: self.create_network_graph(canvas, ax, route))
            create_button.pack(side=tk.LEFT, padx=10, pady=10)

            # Run the Tkinter main loop
            root.mainloop()
    
    #this function is used to simulate the central server sending a list of the participating to all the nodes 
    #Edit this function to allow the server to input dummy nodes into the list
    def sendOutListOfNodes(self):
        print("Sending out list of nodes")
        start = time.time()
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
        print(f"Time taken : {time.time()-start}")
    
    def create_network_graph(self, canvas, ax, route):
        nodes = []
        edges = []
        for partition in route:
            nodes += partition
            for node in range(1,len(partition)):
                edges.append((partition[node-1], partition[node]))

        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)


        # Draw the graph on the specified axes
        pos = {}
        for i in range(len(route)):
            for j in range(len(route[i])):
                pos[route[i][j]] = (j, -i)

        nx.draw(G, pos, ax=ax, with_labels=True, node_color="skyblue", node_size=700)
        canvas.draw()