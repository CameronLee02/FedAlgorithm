# Custom Federated Learning - Series Based Learning
Forked Repository from SIA project of FL to custom algorithm on Federated Learning.

# Three main file for Alian's code (Now in 'Legacy Code' folder):

main_fed_sequential.py: No Homomorphic Encryption, series based custom algorithm (MNIST SUPPORT ONLY) <br> 
main_fed_sequential_threaded.py: No Homomorphic Encryption, series based custom algorithm (THREADING ENABLED) (MNIST SUPPORT ONLY) <br>
main_fed_sequential_threaded_ckks_updated.py: CKKS Partial Addition Homomorphic Encryption, series based custom algorithm (THREADING ENABLED) <br>

# Four main files for Cameron's code (extension from Alian's):

main.py: Contains code necessary to run simulation
network_node.py: Conatins the NetworkSimulationClass which acts as the network in this simulation and transmits messages between the Clients and the Central Server
client_node.py: Contains the ClientNodeClass which acts as a Client in the simulation
server_node.py: Contains the ServerNodeClass which acts as the Central Server in the simulation

## For Speed purposes:

To speed up the simulation process, use the MNIST dataset and change the line of code below

Within the code of the file, there is an import line, 
```
from utils.dataset import get_dataset, exp_details
```
Changing from that into
```
from utils.dataset_limit import get_dataset, exp_details
```
will allow you to create and limit the train size and test size to a more suitable value to see how the simulation runs. (SPECIFIC TO MNIST ONLY)

For example, using dataset_limit with a limit of 1000 sample will have its simulation completed in around 3 minutes, whilst the regular line will utilise the entire dataset and will take approximately 45 minutes to complete. (Values taken from MacBook Pro (M1 Pro, 16 GB Unified Memory)


# IMPORTANT

Python TenSEAL library required for CKKS file, current setup uses Python Version 3.9 to properly install all requirements. More up to date python versions may cause TenSEAL library to not install properly. Requirements.txt states to use numpy 2.0.1. If this doesn't work, use numpy 1.26.4


# Implementation

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs), `--partition_sizes` (minimum size of the partitions) to see how the attack performance changes. For `MNIST` dataset, we set `--model=cnn`.<br>
`--partition_sizes` is not a valid argument parameter in Alian's implementation, and must be removed if you wish to run his legacy code

## For MNIST
```python
python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=5 --local_ep=5 --partition_size=3
```

## For CIFAR-10
```python
python main.py --dataset=CIFAR10 --model=cnn --alpha=1 --num_users=5 --local_ep=5 --partition_size=3
```
