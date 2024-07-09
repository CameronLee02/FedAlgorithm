import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch

'''def test_fun(net_g, datatest, args): #ORIGINAL CODE
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss'''


'''def test_fun(net, dataset, args): #Fix 2: Change the function signature PRIOR WED 10 JUL 12:45 AM
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(dataset):
            data, target = data.to(args.device), target.to(args.device)
            output = net(data)
            print(f'Output shape: {output.shape}')  # Print the shape of the output
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # Ensure output has correct dimensions
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataset.dataset)
    accuracy = 100. * correct / len(dataset.dataset)
    return accuracy, test_loss'''

def test_fun(net, dataset, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.local_bs, shuffle=False)
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = net(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy, test_loss