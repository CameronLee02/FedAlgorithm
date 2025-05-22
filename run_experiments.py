import os


#os.system("python standard_fl_implementation.py --dataset=MNIST --model=cnn --alpha=0.5 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_standard_fl_alpha_0_5")
os.system("python standard_fl_implementation.py --dataset=MNIST --model=cnn --alpha=10 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_standard_fl_alpha_10")
os.system("python main.py --dataset=MNIST --model=cnn --alpha=10 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_alpha_10")
#os.system("python main.py --dataset=MNIST --model=cnn --alpha=0.1 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_alpha_0_1")

"""
os.system("python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=15 --local_ep=5 --partition_size=15 --output_directory=MNIST_baseline_nodes_15")
os.system("python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=20 --local_ep=5 --partition_size=20 --output_directory=MNIST_baseline_nodes_20")

os.system("python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=5 --output_directory=MNIST_baseline_partition_2")
os.system("python main.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=3 --output_directory=MNIST_baseline_partition_3")
os.system("python standard_fl_implementation.py --dataset=MNIST --model=cnn --alpha=1 --num_users=10 --local_ep=5 --output_directory=MNIST_baseline_standard_fl")

os.system("python main.py --dataset=CIFAR10 --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=10 --output_directory=CIFAR_baseline")
os.system("python main.py --dataset=SVHN --model=cnn --alpha=1 --num_users=10 --local_ep=5 --partition_size=10 --output_directory=SVHN_baseline")

os.system("python standard_fl_implementation.py --dataset=CIFAR10 --model=cnn --alpha=1 --num_users=10 --local_ep=5 --output_directory=CIFAR10_baseline_standard_fl")
os.system("python standard_fl_implementation.py --dataset=SVHN --model=cnn --alpha=1 --num_users=10 --local_ep=5 --output_directory=SVHN_baseline_standard_fl")"""
