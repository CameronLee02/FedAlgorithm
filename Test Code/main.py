from network_simluation import NetworkClass
from server_node import ServerNodeClass
from client_node import ClientNodeClass
import sys, getopt
import time

def collectCommandArgs(argv):
    try:
        opts, args = getopt.getopt(argv,"n:", ["num_node="])
    except getopt.GetoptError:
        print('Usage: main.py -n <Number of Nodes>')
        print('Or')
        print('Usage: main.py --num_nodes <Number of Nodes>')
        sys.exit(2)

    num_nodes = 0

    if (len(argv) != 2 or len(opts) == 0):
        print('Usage: main.py -n <Number of Nodes>')
        print('Or')
        print('Usage: main.py --num_nodes <Number of Nodes>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--num_node"):
            try:
                num_nodes = int(arg)
            except ValueError:
                print("Error: -num_node argument must be an integer.")
                sys.exit(2)

    if num_nodes == 0:
        print('Usage: main.py --num_nodes <Number of Nodes>')
        sys.exit(2)

    return num_nodes

if __name__=="__main__":
    num_nodes = collectCommandArgs(sys.argv[1:]) 
    network_instance = NetworkClass()
    
    #add desired number of nodes to network
    for num in range(num_nodes):
        new_node = ClientNodeClass(num+1, network_instance, False)
        network_instance.addNode(new_node)

    #add central server node to network
    central_server = ServerNodeClass(0, network_instance, True)
    network_instance.addNode(central_server)

    #list nodes added to network 
    nodes = network_instance.getNodes()
    for node in nodes.keys():
        print('Node ID:', node, '... object:', nodes[node])

    print("Starting Simulation")
    network_instance.startAllNodes()
    central_server.sendOutListOfNodes()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C 
        central_server.stop()
        print("Stopping Simultaion")
        sys.exit(0)

