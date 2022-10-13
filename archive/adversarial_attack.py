from torch import nn
import torch 
import torchvision.transforms as transforms
import archive.leNet as leNet
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from os.path import exists

# In this file, we compute adversarial examples for pictures in the 
# MNIST dataset using PGD (projected gradient descent)

eps = .01                                   #size of admissible perturbations

pic = ImageOps.grayscale(Image.open("4_pic.png"))
pic = pic.resize((28,28))

preprocess = transforms.Compose([
   transforms.Resize(28),
   transforms.ToTensor(),
])
pic_tensor = preprocess(pic)[None,:,:,:]


#display picture
plt.imshow(pic_tensor[0].numpy().transpose(1,2,0), cmap = 'Greys')
plt.show(block=True)                         


#initialize model
model = leNet.leNet()                               
path_leNet =  './weights_leNet.pth'

if exists(path_leNet): 
    print("Loading pretrained model")
    model.load_state_dict(torch.load(path_leNet))
else:
    print("Did not find pretrained model. Training a model instead.")
    model = leNet.train_leNet(4, 2)
    torch.save(model.state_dict(), path_leNet)

model.eval()       


#Finding the adversarial perturbation
delta = torch.zeros_like(pic_tensor, requires_grad=True)
opt = torch.optim.SGD([delta], lr=1e-1)    
#run PGD for 30 steps 
for t in range(50):
    pred = model(pic_tensor + delta)
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([3]))
    if t % 5 == 0:
        print(t, loss.item())
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(-eps, eps)     #project deta to the l^∞ open ball of radius eps around 0


#output results
pred = model(pic_tensor + delta)
max_class = pred.max(dim=1)[1].item()     #class predicted for the last picture 

print("Norm of perturbation is ", delta.max().item())
print("Predicted digit", max_class)
print("Confidence:", nn.Softmax(dim=1)(pred)[0, max_class].item())

plt.imshow((pic_tensor +delta)[0].detach().numpy().transpose(1,2,0), cmap='Greys')
plt.show(block=True)


#Now test how the leNetVariant model performs on this adversarial example

#initialize model
model = leNet.leNetVariant()                               
path_leNetVariant =  './weights_leNetVariant.pth'

if exists(path_leNetVariant): 
    print("Loading pretrained model")
    model.load_state_dict(torch.load(path_leNetVariant))
else:
    print("Did not find pretrained model. Training a model instead.")
    model = leNet.train_leNetVariant(4, 2)
    torch.save(model.state_dict(), path_leNetVariant)

model.eval() 

pred = model(pic_tensor)
print(pred)
max_class = pred.max(dim=1)[1].item()
print("Predicted digit", max_class)
print("Confidence:", nn.Softmax(dim=1)(pred)[0, max_class].item())