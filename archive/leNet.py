from torch import nn
import torch 
import torchvision 
import torchvision.transforms as transforms

# In this file we define the (standard) LeNet model and train it on the MNIST dataset

class leNet(nn.Module):
    def __init__(self):
        super(leNet, self).__init__()
        self.conv_layers = nn.Sequential( 
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 120, 4),
            nn.ReLU(),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.ReLU(),
        )

    
    def forward(self, x):
        x = self.conv_layers(x)
        y = x.view((x.shape[0], x.shape[1]))   #flatten x 
        return self.fully_connected_layers(y)

class leNetVariant(nn.Module):
    def __init__(self):
        super(leNetVariant, self).__init__()
        self.conv_layers = nn.Sequential( 
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 120, 4),
            nn.ReLU(),
        )
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(120, 10),
            nn.ReLU(),
        )

    
    def forward(self, x):
        x = self.conv_layers(x)
        y = x.view((x.shape[0], x.shape[1]))   #flatten x 
        return self.fully_connected_layers(y)

#This function trains a leNet model for n_epochs using mini-batch gradient descent with batches of size batch_size
def train_leNet(batch_size, n_epochs): 
    #batch_size needs to be at least 2 otherwise training breaks 

    #prepare data
    transform = transforms.ToTensor()
    training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    #initialize model, loss and optimiser
    model = leNet()
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = .001, momentum = .9)

    #train model using mini-batch gradient descent 
    for epoch in range(0, n_epochs):
        running_loss = 0
        for i, data in enumerate(loader, 0):
            optimizer.zero_grad()

            pt, lbl = data

            output = model.forward(pt)

            loss = loss_fn(output, lbl.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0 

    print('Finished Training')

    #output model
    return model

#This function trains a leNetVariant model for n_epochs using mini-batch gradient descent with batches of size batch_size
def train_leNetVariant(batch_size, n_epochs): 
    #batch_size needs to be at least 2 otherwise training breaks 

    #prepare data
    transform = transforms.ToTensor()
    training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)

    #initialize model, loss and optimiser
    model = leNetVariant()
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = .001, momentum = .9)

    #train model using mini-batch gradient descent 
    for epoch in range(0, n_epochs):
        running_loss = 0
        for i, data in enumerate(loader, 0):
            optimizer.zero_grad()

            pt, lbl = data

            output = model.forward(pt)

            loss = loss_fn(output, lbl.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0 

    print('Finished Training')

    #output model
    return model

