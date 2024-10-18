from network_simluation import NetworkClass
from server_node import ServerNodeClass
from client_node import ClientNodeClass
import sys, getopt
import time

def collectCommandArgs(argv):
    try:
        opts, args = getopt.getopt(argv,"n:t:", ["num_node=", "type=="])
    except getopt.GetoptError:
        print('Usage: main.py -n <Number of Nodes> -t <type of test>')
        print('Or')
        print('Usage: main.py --num_nodes <Number of Nodes> --type <type of test>')
        sys.exit(2)

    num_nodes = 0
    test_type = ""

    if (len(argv) != 4 or len(opts) == 0):
        print('Usage: main.py -n <Number of Nodes> -t <type of test>')
        print('Or')
        print('Usage: main.py --num_nodes <Number of Nodes> --type <type of test>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-n", "--num_node"):
            try:
                num_nodes = int(arg)
            except ValueError:
                print("Error: invalid --num_nodes argument")
                sys.exit(2)
        if opt in ("-t", "--type"):
            try:
                test_type = str(arg)
            except ValueError:
                print("Error: invalid --type argument")
                sys.exit(2)

    if num_nodes == 0 or test_type == "":
        print('Usage: main.py --num_nodes <Number of Nodes> --type <type of test>')
        sys.exit(2)

    return num_nodes, test_type

if __name__=="__main__":
    num_nodes, test_type = collectCommandArgs(sys.argv[1:]) 
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

    if test_type == "pow":
        print("Starting Simulation to test PoW")
    elif test_type == "route":
        print("Starting Simulation to test in-house route calc")
    else:
        print("Error: invalid --type argument")
        sys.exit(2)
        
    network_instance.startAllNodes()
    central_server.sendOutListOfNodes(test_type)


    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Handle Ctrl+C 
        central_server.stop()
        print("Stopping Simultaion")
        sys.exit(0)

