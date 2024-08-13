import torch
from torch.utils.data import DataLoader, Dataset

class LocalUpdate:
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.dataset = dataset
        self.idxs = list(idxs)
        self.ldr_train = DataLoader(DatasetSplit(self.dataset, self.idxs),
                                    batch_size=self.args.local_bs, shuffle=True)

    def train(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.5)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs = net(images)
                loss = torch.nn.functional.cross_entropy(log_probs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)  # Adjust max_norm for balance

                optimizer.step()
                batch_loss.append(loss.item())

            # Log at the end of each epoch instead of every batch
            epoch_loss_avg = sum(batch_loss) / len(batch_loss)
            print(f'Epoch {iter} completed with average loss: {epoch_loss_avg:.4f}')
            epoch_loss.append(epoch_loss_avg)
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
