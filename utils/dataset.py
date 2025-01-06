import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from utils.sampling import sample_dirichlet_train_data, synthetic_iid

def get_dataset(args):
    # CIFAR-10: 10 classes, 60,000 images (50,000 train, 10,000 test)
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

        # Sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(
            train_dataset, args.num_users+1, args.num_samples, args.alpha)

    # MNIST: 10 classes, 60,000 training examples, 10,000 testing examples.
    elif args.dataset == 'MNIST':
        data_dir = './data/mnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        # Sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(
            train_dataset, args.num_users+1, args.num_samples, args.alpha)
    
    # SVHN: Street View House Numbers dataset
    elif args.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ])
        train_dataset = datasets.SVHN(root='./data/svhn', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data/svhn', split='test', download=True, transform=transform)

        # Sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(
            train_dataset, args.num_users+1, args.num_samples, args.alpha)

    # Synthetic dataset: 10 classes, 100,000 examples.
    elif args.dataset == 'Synthetic' and args.iid:
        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))

        # Sample iid data
        dict_party_user, dict_sample_user = synthetic_iid(train_dataset, args.num_users, args.num_samples)

    elif args.dataset == 'Synthetic' and not args.iid:
        data_dir = './data/synthetic/synthetic_x_0.npz'
        synt_0 = np.load(data_dir)
        X = synt_0['x'].astype(np.float64)
        Y = synt_0['y'].astype(np.int32)

        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

        train_dataset = DataLoader(TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()))
        test_dataset = DataLoader(TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()))

        # Sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(
            train_dataset, args.num_users, args.num_samples, args.alpha)

    # Example FLAIR dataset loading (Modify as needed for your specific FLAIR dataset)
    elif args.dataset == 'FLAIR':
        from nibabel import load as load_nii

        class FLAIRDataset(Dataset):
            def __init__(self, data_dir, transform=None):
                self.data_dir = data_dir
                self.transform = transform
                self.image_files = [f for f in os.listdir(data_dir) if f.endswith('.nii')]

            def __len__(self):
                return len(self.image_files)

            def __getitem__(self, idx):
                img_name = os.path.join(self.data_dir, self.image_files[idx])
                image = load_nii(img_name).get_fdata()
                image = torch.tensor(image, dtype=torch.float32)
                if self.transform:
                    image = self.transform(image)
                return image, 0  # Assuming binary classification, otherwise change the label

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        train_dataset = FLAIRDataset(data_dir='./data/flair/train', transform=transform)
        test_dataset = FLAIRDataset(data_dir='./data/flair/test', transform=transform)

        # Sample non-iid data
        dict_party_user, dict_sample_user = sample_dirichlet_train_data(
            train_dataset, args.num_users, args.num_samples, args.alpha)

    else:
        train_dataset = []
        test_dataset = []
        dict_party_user, dict_sample_user = {}, {}
        print('+' * 10 + 'Error: unrecognized dataset' + '+' * 10)

    return train_dataset, test_dataset, dict_party_user, dict_sample_user

def exp_details(args):
    print('\nExperimental details:')
    print(f'Model     : {args.model}')
    print(f'Optimizer : sgd')
    print(f'Learning rate: {args.lr}')
    print(f'Global Rounds: {args.epochs}\n')

    print('Federated parameters:')
    print('{} dataset, '.format(args.dataset) + f' has {args.num_classes} classes')

    if not args.iid:
        print(f'Level of non-iid data distribution: \u03B1 = {args.alpha}')
    else:
        print('The training data are iid across parties')
    print(f'Number of users    : {args.num_users}')
    print(f'Local Batch size   : {args.local_bs}')
    print(f'Local Epochs       : {args.local_ep}\n')

    return
