from DenseNet import DenseNet
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import csv
import pandas as pd

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():


    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    

    
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    dn = DenseNet(growth_rate=32, num_init_features=64, bn_size=4, drop_rate=0, num_classes=10, memory_efficient=False)

    model_on_gpu = dn.to(device)



    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_on_gpu.parameters(), lr=0.001, momentum=0.9)

    accuracy_matrix = np.matrix([['epoch','accuracy']])
    loss_matrix = np.matrix([['epoch', 'loss']])
    # accuracy_matrix = np.append(accuracy_matrix,np.array([[1,2]]), axis=0)

    for epoch in range(50):  # loop over the dataset multiple times

        correct = 0
        total_for_acc = 0
        running_loss = 0.0
        total_loss = 0.0
        total_count = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_on_gpu(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                print(running_loss/2000)
                total_loss += running_loss
                total_count += 1
                running_loss = 0.0

        loss_matrix = np.append(loss_matrix, np.array([[epoch,total_loss/(2000*total_count)]]), axis=0)
        total_count = 0
        total_loss = 0.0    

        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_on_gpu(inputs)
            _, predicted = torch.max(outputs,1)

            correct += (predicted == labels).sum().item()
            total_for_acc += labels.size(0)
        accuracy_matrix = np.append(accuracy_matrix, np.array([[epoch, correct / total_for_acc]]),axis=0)
        print(correct/total_for_acc)

        correct = 0
        incorrect = 0 

    DF = pd.DataFrame(loss_matrix) 
    DF.to_csv("Metric_Loss.csv", sep=',', header=False, index=False)

    DF2 = pd.DataFrame(accuracy_matrix)
    DF2.to_csv("Metric_Accuracy.csv", sep=',', header=False, index=False)
    torch.save(dn.state_dict(), 'Finished_Model.pt')

        #Later to restore:
        

    print('Finished Training')

    
if __name__ == '__main__':
    main()