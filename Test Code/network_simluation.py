import threading

class NetworkClass:
    def __init__(self):
        self.nodes = {}
        self.fake_node_list = []
        self.server_node = None

    def addNode(self, node):
        if node.node_id == 0:
            self.server_node = node
        else:
            self.nodes[node.node_id] = node
    
    def getNodes(self):
        return self.nodes
    
    def messageSingleNode(self, sender_id, receiver_id ,message):
        if (sender_id in self.nodes.keys() or sender_id in self.fake_node_list) and receiver_id in self.nodes.keys():
            print(f"sending message to {receiver_id}")
            self.nodes[receiver_id].receiveMessage(sender_id, message)
        elif (sender_id in self.nodes.keys() or sender_id in self.fake_node_list) and receiver_id in self.fake_node_list:
            print(f"sending message to {receiver_id}")
            self.server_node.receiveMessage(sender_id, message, receiver_id)
    
    #this function is used to send a message to all the node, except the central server node.
    def messageAllNodesExcludeServer(self, sender_id, message):
        threads = []
        for key, value in self.nodes.items(): #sends the message to all real nodes
            if key != sender_id:
                t = threading.Thread(target=value.receiveMessage, args=(sender_id, message))
                t.start()
                threads.append(t)
        
        for node in self.fake_node_list: #sends the message to all fake nodes
            if node != sender_id:
                t = threading.Thread(target=self.server_node.receiveMessage, args=(sender_id, message, node))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
    
    def startAllNodes(self):
        for node in self.nodes.values():
            node.daemon = True
            node.start()
        print("started up all the client nodes")
        
        self.server_node.daemon = True
        self.server_node.start()
        self.server_node.getNodeList()
        self.fake_node_list = self.server_node.getFakeListOfNodes()
        print("started up the Server Node")
        print(f"Real Nodes: {list(self.nodes.keys())}")
        print(f"Dummy Nodes: {self.fake_node_list}")

