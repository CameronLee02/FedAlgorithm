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
    
    #this function is used to simulate the central server sending a list of the participating to all the nodes 
    def sendOutListOfNodes(self):
        print("Sending out list of nodes")
        start = time.time()
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
        print(f"Time taken : {time.time()-start}")