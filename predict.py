import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from model import Model
from Dataset import ToTorchFormatTensor, CustomDataset

def plot(images, labels, true):
    f, ax = plt.subplots(4,8)
    count = 0
    for i in range(4):
        for j in range(8):
            ax[i, j].imshow(images[count,0,:,:], cmap="gray")
            ax[i, j].set_title(f"Predicted:{labels[count]}, Actual:{true[count]}")
            ax[i, j].axis('off')
            count += 1 
    plt.show()

def predict(model, val_loader):
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            images = images.cuda()
            targets = targets.cuda()
            output  = model(images)
            y_score =  torch.topk(output,1).indices.reshape(images.size(0)).detach().cpu().numpy()
            plot(images.detach().cpu().numpy(),y_score , targets.detach().cpu())
            break
            

if __name__ == '__main__':
    data_transform = torchvision.transforms.Compose([ToTorchFormatTensor()])
    test_generator = CustomDataset('Dataset/t10k-images.idx3-ubyte', 'Dataset/t10k-labels.idx1-ubyte',transform = data_transform, phase = 'test')
    test_loader = torch.utils.data.DataLoader(test_generator, batch_size= 1000, shuffle = True)

    model = Model()

    checkpnt = torch.load('checkpoints/checkpoint.pt')
    model.load_state_dict(checkpnt['model_state_dict'])
    print("weights loaded")
    model.cuda()
    predict(model, test_loader)

