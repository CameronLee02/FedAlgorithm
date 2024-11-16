import threading

class NetworkClass:
    def __init__(self):
        self.nodes = {}
        self.server_node = None
        self.pow_volunteer = None
        self.pow_volunteer_lock = threading.Lock()

    #Adds a node to the network. ID:0 is reserved for the central server
    def addNode(self, node):
        if node.node_id == 0:
            self.server_node = node
        else:
            self.nodes[node.node_id] = node
    
    def getNodes(self):
        return self.nodes
    
    def getPoWVolunteer(self):
        return self.pow_volunteer
    
    def getPoWVolunteerLock(self):
        return self.pow_volunteer_lock
    
    def setPoWVolunteer(self, volunteer):
        self.pow_volunteer = volunteer
    
    #this function is used to send 1 message to 1 node
    def messageSingleNode(self, sender_id, receiver_id ,message):
        if sender_id in self.nodes.keys() and receiver_id in self.nodes.keys():
            self.nodes[receiver_id].receiveMessage(sender_id, message)
    
    #This function is used to send a message to just the central server node
    def messageCentralServer(self, sender_id, message):
        self.server_node.receiveMessage(sender_id, message)
    
    #this function is used to send a message to all the node, except the central server node.
    def messageAllNodesExcludeServer(self, sender_id, message):
        threads = []
        for key, value in self.nodes.items():
            if key != sender_id:
                t = threading.Thread(target=value.receiveMessage, args=(sender_id, message))
                t.start()
                threads.append(t)

        for t in threads:
            t.join()
