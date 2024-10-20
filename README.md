# Custom Federated Learning - Series Based Learning
Forked Repository from SIA project of FL to custom algorithm on Federated Learning.

# Three main file:

main_fed_sequential.py: No Homomorphic Encryption, series based custom algorithm (MNIST SUPPORT ONLY) <br> 
main_fed_sequential_threaded.py: No Homomorphic Encryption, series based custom algorithm (THREADING ENABLED) (MNIST SUPPORT ONLY) <br>
main_fed_sequential_threaded_ckks_updated.py: CKKS Partial Addition Homomorphic Encryption, series based custom algorithm (THREADING ENABLED) <br>

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

Python TenSEAL library required for CKKS file, current setup uses Python Version 3.9 to properly install all requirements. More up to date python versions may cause TenSEAL library to not install properly.


# Implementation

You can try different `--alpha` (data distribution), `--num_users`(number of parties), `--local_ep` (number of local epochs) to see how the attack performance changes. For `MNIST` dataset, we set `--model=cnn`.

## For MNIST
```python
python main_fed_sequential_threaded_ckks_updated.py --dataset=MNIST --model=cnn --alpha=1 --num_users=6 --local_ep=5
```

## For CIFAR-10
```python
python main_fed_sequential_threaded_ckks_updated.py --dataset=CIFAR10 --model=cnn --alpha=1 --num_users=6 --local_ep=5
```
