"""
CNN with 3 conv layers and a fully connected classification layer
PATTERN RECOGNITION EXERCISE:
Fix the three lines below marked with PR_FILL_HERE
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Grayscale
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

class Flatten(nn.Module): #nn.Module is ancestor
    """
    Flatten a convolution block (=convolution layer) into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1) #28x28image--> 1D tensor with length 784
        return x


class PR_CNN(nn.Module):
    """
    Simple feed forward convolutional neural network

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, batch_size: int, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_CNN, self).__init__() #to make sure this class extends the nn.Module

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (batch_size, 1, 28, 28) #batch size, colour channels, height, width

        # First layer
        self.conv1 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels (nr of neurons/kernels) and the kernel size

            nn.Conv2d(in_channels=1, out_channels=24, kernel_size=3, stride=3),#input channels depends on nr of colour channels; would be 3 in case of RGB image
            #filter tensors are not directly visible, but: Our tensors are rank-4 tensors. The first axis represents the
            # number of filters. The second axis represents the depth of each filter which corresponds to the number of
            # input channels being convolved.The last two axes represent the height and width of each filter. We can pull
            # out any single filter by indexing into the weight tensor’s first axis.
            ##kernel size should be small/big enough to detect the lines. Only kernel size=3 possible with such a small
            #input image if we want to have 3 hidden layers (otherwise image will be smaller than the kernel size
            #for the last hidden layer, which is not allowed of course
            #output size = [(input size + 2padding size -kernel size) / stride + 1] , eg here in first layer:
            #(28-3)/3+1=9, so the image is shrinked to 9x9
            nn.LeakyReLU(0.1)
        )

        # Second layer
        self.conv2 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels (nr of neurons/kernels) and the kernel size
            nn.Conv2d(in_channels=24, out_channels=192, kernel_size=3, stride=3),
            # input channels would be 3 in case of RGB image
            nn.LeakyReLU(0.1)
        )

        # Third layer
        self.conv3 = nn.Sequential(
            # PR_FILL_HERE: Here you have to put the input channels, output channels (nr of neurons/kernels) and the kernel size
            nn.Conv2d(in_channels=192, out_channels=1536, kernel_size=3, stride=3),
            # input channels would be 3 in case of RGB image
            nn.LeakyReLU(0.1)
        )

        # Classification layer /fully connected layer (--> fc)
        self.fc = nn.Sequential(
            Flatten(), #need to flatten when passing an output tensor from a convolutional layer to a linear layer
            # PR_FILL_HERE: Here you have to put the output size of the linear layer. DO NOT change 1536!
            nn.Linear(in_features=1536, out_features=10) #tensor: rank-2 with height (length of desired output features)
            # and width (length of input features) axes.--> matrix multiplication tensor *in features
        )

    def forward(self, x): #x is a tensor
        """
        Computes forward pass on the network (tensor flows forward through the NN)
        the forward method is the actual transformation. Always when building NN or layers, we must implement this
        forward method.
        The goal of the overall transformation is to transform or map the input to the correct prediction output class,
        and during the training process, the layer weights (data) are updated in such a way that cause the mapping to
        adjust to make the output closer to the correct prediction.

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        #input layer would be:
        #x=x
        x = self.conv1(x) #first hidden layer #dont call the forward method explicitly, cause: forward method is executed
        # automatically through __call__ (implemented in Pytorch module class)
        x=self.conv2(x)
        x=self.conv3(x)

        #pooling
        #x=nn.functional.max_pool2d(x, kernel_size=2, stride=2) #try with 2 as my images are quite small (28x28)
        #our images are too small for pooling after the 3 hidden layers


        x = self.fc(x) #classification layer
        return x

class im_object:
    def __init__(self, values):
        self.label = values[0]
        self.im = np.reshape(values[1:],((28,28)))

def get_num_correct_pred(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item() #this gives the # of times the most likely class is the true class

def train_and_test (network, train_loader, test_loader, nr_of_epochs, learning_rate,batch_size):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    training_losses=[]
    training_accuracies=[]
    test_losses = []
    test_accuracies = []

    for epoch in range(nr_of_epochs):
        total_loss=0
        total_correct=0
        total=0

        for batch in train_loader: #get batch; batch is a sequence with length 2 (images & labels)
            images, labels = batch  # images is a single tensor with a shape that reflects the following axes:
            # (batch size, input channels, height, width)

            #now pass the batch of images to the network
            predictions=network(images) #returns a tensor of size (batch size, nr of prediction classes), i.e. the probability
            #for each class for each image
            loss=nn.functional.cross_entropy(predictions, labels) #calculate loss

            optimizer.zero_grad() #zero out gradient attributes of our network's parameters
            loss.backward() #calculate gradients & add to the grad attributes of our network's parameter
            optimizer.step() # Updating the weights
            #correct preds should now be larger for the same batch

            total+=1
            total_loss+=loss.item()
            total_correct+=get_num_correct_pred(predictions, labels)
        average_loss=total_loss/(total*batch_size)
        average_accuracy=total_correct/(total*batch_size)*100
        training_losses.append(average_loss)
        training_accuracies.append(average_accuracy)

        average_loss,average_accuracy=test(network, test_loader, batch_size)
        test_losses.append(average_loss)
        test_accuracies.append(average_accuracy)


    return training_losses, training_accuracies, test_losses, test_accuracies

@torch.no_grad()#to omit gradient tracking, which would use memory and is only necessery during training
def test(network, test_loader, batch_size):
    total_loss = 0
    total_correct = 0
    total = 0

    for batch in test_loader:  # get batch; batch is a sequence with length 2 (images & labels)
        images, labels = batch  # images is a single tensor with a shape that reflects the following axes:
        # (batch size, input channels, height, width)

        # now pass the batch of images to the network
        predictions = network(
            images)  # returns a tensor of size (batch size, nr of prediction classes), i.e. the probability
        # for each class for each image
        loss = nn.functional.cross_entropy(predictions, labels)  # calculate loss

        total += 1
        total_loss += loss.item()
        total_correct += get_num_correct_pred(predictions, labels)
    average_loss = total_loss / (total * batch_size)
    average_accuracy = total_correct / (total * batch_size) * 100
    return average_loss, average_accuracy

def create_plot(training_losses, training_accuracies,test_losses, test_accuracies, nr_of_epochs, i):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(np.arange(nr_of_epochs), training_losses, color="green", label="training loss")
    plt.plot(np.arange(nr_of_epochs), test_losses, color="red", label="test loss")
    plt.legend(loc='best')
    plt.subplot(1,2,2)
    plt.plot(np.arange(nr_of_epochs), training_accuracies, color="green", label="training accuracy")
    plt.plot(np.arange(nr_of_epochs), test_accuracies, color="red", label="test accuracy")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("Model" + str(i) + ".png", bbox_inches='tight')

def load_dataset (batch_size):
    ####load the full MNIST dataset####
    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = torchvision.datasets.ImageFolder(root='mnist-permutated-png-format/mnist/train',
                                                     transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='mnist-permutated-png-format/mnist/test', transform=transform)
    # load the sets to the dataloader to be able to process the sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def load_permutated_dataset(batch_size):
    transform = Compose([Grayscale(num_output_channels=1), ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])
    train_dataset = torchvision.datasets.ImageFolder(root='mnist-permutated-png-format/mnist/train', transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root='mnist-permutated-png-format/mnist/test', transform=transform)
    # load the sets to the dataloader to be able to process the sets
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader




def optimization(batch_sizes, nr_of_epochs, learning_rates):
    for batch_size in batch_sizes:
        train_loader, test_loader=load_dataset(batch_size)
        #create instance of our CNN
        network=PR_CNN(batch_size)
        #test different learning rates
        for learning_rate in learning_rates:
            print("batch size=", batch_size, "learning rate=", learning_rate)
            training_losses, training_accuracies, test_losses, test_accuracies=train_and_test(network, train_loader, test_loader, nr_of_epochs, learning_rate, batch_size)
            return training_losses, test_losses, training_accuracies, test_accuracies

#main
find_optimal=False
if find_optimal==True:
    batch_sizes=[50,100, 500] #whatever
    nr_of_epochs=12
    learning_rates=[0.001, 0.01,0.1]
    training_losses, test_losses, training_accuracies, test_accuracies=optimization(batch_sizes, nr_of_epochs, learning_rates)
#we found:
#optimal parameters: learning rate=0.001, batch size =whatever, slope=0.1, epochs: after 4-6, the accuracy
#decreases for the first time; thereafter, it increases and decreases, but stays roughly the same.

batch_size=100
learning_rate=0.001
nr_of_epochs=12

#load dataset
permutated=True #set to True to run with the permutated dataset
if permutated==False:
    train_loader, test_loader=load_dataset(batch_size)
else:
    train_loader, test_loader=load_permutated_dataset(batch_size)

iterations=[1]
try_different_initializations=False
if try_different_initializations==True: #try different initializations (there is no big difference)
    iterations=range(0,5) #or as many models you wish to create
for i in iterations:
    network=PR_CNN(batch_size) #create instance of our CNN
    parameters=list(network.conv1.parameters())
    file = open(f'CNN_initializing{i}', 'w+')
    file.write(f'model number {i} \n')
    file.write(f'weights {parameters[0]}, biases {parameters[1]} \n')
    training_losses, training_accuracies, test_losses, test_accuracies = train_and_test(network, train_loader, test_loader,nr_of_epochs, learning_rate,batch_size)
    create_plot(training_losses, training_accuracies, test_losses, test_accuracies, nr_of_epochs, i)
    #train_accuracy=training_accuracies[-1]
    #test_accuracy=test_accuracies[-1]
    file.write(f'training accuracies {training_accuracies}, testing accuracies {test_accuracies} \n')
    file.close()





