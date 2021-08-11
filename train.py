import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from model import Model
from Dataset import ToTorchFormatTensor, CustomDataset


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def validate(model, val_loader, criterion):
    val_loss =AverageMeter()
    val_accuracy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.cuda()
            targets = targets.cuda()
            output  = model(images)
            loss = criterion(output, targets)
            
            val_loss.update(loss.item(), images.size(0))

            y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
            y_true = targets.detach().cpu().numpy()
            acc = accuracy_score(y_true, y_score)
            val_accuracy.update(acc, images.size(0))

    return val_loss.avg, val_accuracy.avg


def train(model, train_loader, epochs, criterion, optimizer):
  
    model.train()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        output  = model(images)
        loss = criterion(output, targets)
        train_loss.update(loss.item(), images.size(0))
        loss.backward()
        optimizer.step()

        y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
        y_true = targets.detach().cpu().numpy()
        acc = accuracy_score(y_true, y_score)
        train_accuracy.update(acc, images.size(0))
    return train_loss.avg, train_accuracy.avg

if __name__ == '__main__':

    data_transform = torchvision.transforms.Compose([ToTorchFormatTensor()])
    train_generator = CustomDataset('Dataset/train-images.idx3-ubyte', 'Dataset/train-labels.idx1-ubyte',transform = data_transform, phase = 'train')
    val_generator = CustomDataset('Dataset/train-images.idx3-ubyte', 'Dataset/train-labels.idx1-ubyte',transform = data_transform, phase = 'val')
    test_generator = CustomDataset('Dataset/t10k-images.idx3-ubyte', 'Dataset/t10k-labels.idx1-ubyte',transform = data_transform, phase = 'test')

    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= 1000, shuffle = True)
    val_loader = torch.utils.data.DataLoader(train_generator, batch_size= 1000, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_generator, batch_size= 1000, shuffle = True)

    model = Model()
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 15

    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []


    print('starting trainig')
    for epoch in range(epochs):
        loss_train, accuracy_train =train(model, train_loader, epochs, criterion,optimizer)
        loss_val, accuracy_val = validate(model, val_loader, criterion)
        print(f'Epoch:{epoch} --> train_loss: {loss_train:.3f},  train_accuracy: {accuracy_train:.3f}, val_loss: {loss_val:.3f},  val_accuracy: {accuracy_val:.3f}')
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        train_accuracy.append(accuracy_train)
        val_accuracy.append(accuracy_val)

    loss_test, accuracy_test = validate(model, test_loader, criterion)

    torch.save({
        'model_state_dict': model.state_dict(),
            }, f"checkpoints/checkpoint.pt")

    x = [i for i in range(epochs)]
    plt.figure()
    plt.plot(x, train_loss, label = "train_loss")
    plt.plot(x, val_loss, label = "val_loss")
    plt.plot(x, train_accuracy, label = "train_accuracy")
    plt.plot(x, val_accuracy, label = "val_accuracy")
    plt.title(f"test loss: {loss_test}, accuracy: {accuracy_test}")
    plt.xticks(x)
    plt.legend()
    plt.show()