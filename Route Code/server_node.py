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
            self.network.setRouteVolunteer(None)
    
    #this function is used to simulate the central server sending a list of the participating to all the nodes 
    #Edit this function to allow the server to input dummy nodes into the list
    def sendOutListOfNodes(self):
        print("Sending out list of nodes")
        start = time.time()
        self.network.messageAllNodesExcludeServer(0, {"NODE_LIST" : list(self.node_list.keys())})
        print(f"Time taken : {time.time()-start}")