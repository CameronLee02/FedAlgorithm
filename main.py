"""
Sequential Federated Learning Algorithm THREADED WITH CKKS ENCRYPTION (WITH OVERHEAD GUI) and in-house routing 
Cameron Lee - 23247003
Most up to date code. Last Modified: 14-11-2024
Requires 3 other files which hold the necessary custom classes (client_node.py, server_node.py, network_node.py)
"""

from network_node import NetworkSimulationClass
from server_node import ServerNodeClass
from client_node import ClientNodeClass
import torch
from utils.options import args_parser


if __name__=="__main__":
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    network_simulation_instance = NetworkSimulationClass(args)

    #Adds central server to the network
    central_server = ServerNodeClass(0, network_simulation_instance)
    network_simulation_instance.addNode(central_server)

    #Add all the nodes to the network
    for num in range(args.num_users):
        new_node = ClientNodeClass(num+1, network_simulation_instance)
        network_simulation_instance.addNode(new_node)

    nodes = network_simulation_instance.getNodes()
    for node in nodes.keys():
        print('Node ID:', node, '... object:', nodes[node])
        
    central_server.getNodeList(nodes)
    network_simulation_instance.create_gui()

