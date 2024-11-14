from network_node import NetworkSimulationClass
from server_node import ServerNodeClass
from client_node import ClientNodeClass
import torch
from utils.options import args_parser


if __name__=="__main__":
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    simulation_instance = NetworkSimulationClass(args)

    #Adds central server to the network
    central_server = ServerNodeClass(0, simulation_instance)
    simulation_instance.addNode(central_server)

    #Add all the nodes to the network
    for num in range(args.num_users):
        new_node = ClientNodeClass(num+1, simulation_instance)
        simulation_instance.addNode(new_node)

    nodes = simulation_instance.getNodes()
    for node in nodes.keys():
        print('Node ID:', node, '... object:', nodes[node])
        
    central_server.getNodeList(nodes)
    simulation_instance.create_gui()

