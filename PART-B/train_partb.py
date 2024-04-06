import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Subset
from googlenet_pytorch import GoogLeNet
import wandb

hyperparameter_defaults= {
             "batch_norm":0, # Whether to use batch normalization (0 for False, 1 for True)
             "no_of_layers":16, # Number of convolutional layers
             "file_org":1, # Factor by which filter size increases
             "Filter_size":3, # Size of the filters
              "max_epochs":2, # Maximum number of epochs for training
              "batch_size":128, # Batch size for training
              "drop_out":0.3 # Dropout probability
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def train_val_dataset(dataset, val_split=0.1):
    """
    Split the dataset into train and validation subsets.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to be split.
        val_split (float): The fraction of the dataset to include in the validation split (default: 0.1).

    Returns:
        dict: A dictionary containing 'train' and 'val' subsets.
    """
    # Split dataset into train and validation indices
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    # Return a dictionary containing train and validation subsets
    return dict({
        "train" : Subset(dataset, train_idx),
        "val" : Subset(dataset, val_idx)
    })

def createDataLoader():
    """
    Create data loaders for training and validation sets.

    Returns:
        tuple: Tuple containing training and validation data loaders.
    """
    # Define mean and standard deviation values for normalization
    mean_values = (0.5, 0.5, 0.5)
    std_values = (0.5, 0.5, 0.5)
    op_size = 32
    # Define transformations for training set
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(op_size),
        transforms.ToTensor(),
        transforms.Normalize(mean_values,std_values ),
        ])
    # Define transformations for horizontal flipping augmentation

    transform_horizontal = transforms.Compose([
        transforms.RandomResizedCrop(op_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])
    # Define transformation for vertical flipping augmentation

    transform_vertical = transforms.Compose([
        transforms.RandomResizedCrop(op_size), # Random resized crop
        transforms.RandomVerticalFlip(), # Random vertical flip
        transforms.ToTensor(), # Convert to tensor
        ])
    # Define transformation for inverting augmentation
    transform_Invert= transforms.Compose([
        transforms.RandomResizedCrop(op_size), # Random resized crop
        transforms.RandomInvert(), # Random invert
        transforms.ToTensor(), # Convert to tensor
        ])
    # Define the directory containing the dataset
    dataset_url = "/content/drive/MyDrive/dl-assigment-2/inaturalist_12K/train"
    # Load the image data with different transformations
    img_data = torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_train)
    img_data_hori= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_horizontal)
    img_data_vert= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_vertical)
    img_data_inve= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_Invert)
    # Combine all datasets
    img_data = img_data + img_data_inve + img_data_vert + img_data_hori
    # Split the combined dataset into train and validation sets
    img_data = train_val_dataset(img_data)

    # Get train and validation subsets
    X_train=img_data['train']
    X_Valid=img_data['val']

    # Create data loaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(X_train, batch_size=128, shuffle=True)
    validationloader = torch.utils.data.DataLoader(X_Valid, batch_size=128, shuffle=False)

    # Get an iterator for trainloader
    dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # Return trainloader and validationloader
    return trainloader,validationloader


def accuracy(dataset_itr,model,norm_fact):
    """
    Calculate the accuracy of the model on a given dataset.

    Args:
        dataset_itr (iterable): Iterable containing the dataset.
        model (torch.nn.Module): The model to evaluate.
        norm_fact: Normalization factor.

    Returns:
        float: Accuracy of the model on the dataset.
    """
    total = 0
    pred_count = 0
    # Iterate over the dataset
    for dataset in dataset_itr:
        X,y=dataset
        # Move data to the appropriate device
        X = X.to(device)
        y = y.to(device)
        # Get predictions from the model
        pred = torch.max(model(X,norm_fact).data,1)[1]
        # Update counts
        total+=y.size(0)
        pred_count+=(pred==y).sum().item()
        # Calculate accuracy
        acc = (100*pred_count)/total
    return acc

def fit(batch_norm,k,file_org,F,max_epochs,batch_size,d,act_fun=nn.ReLU()):
    """
    Train the model.

    Args:
        batch_norm (bool): Whether to use batch normalization.
        k (int): Filter size.
        file_org (float): Factor by which filter size increases.
        F (int): Filter size for convolutional layers.
        max_epochs (int): Maximum number of epochs for training.
        batch_size (int): Batch size for training.
        d (float): Dropout probability.
        act_fun: Activation function to use.

    Returns:
        None
    """
    # Initialize the model
    model = GoogLeNet.from_pretrained('googlenet')
    model.aux_logits=False
    # Define optimizer and loss function
    opt=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)

    loss_fn = nn.CrossEntropyLoss()
    train_loss,val_loss=[],[]
    # Calculate number of iterations per epoch
    n_iter=np.ceil(8999/batch_size)
    # Create data loaders
    trainloader,validationloader = createDataLoader()
    epoch = 0
    # Start training loop
    while epoch<max_epochs:
        for key,data in enumerate(trainloader,0):
            X,y=data
            X = X.to(device)
            y = y.to(device)
            # Zero gradients
            opt.zero_grad()

            outputs=model(X,batch_norm)
            loss=loss_fn(outputs,y)
            loss.backward()
            opt.step()

            del X,y,outputs
            torch.cuda.empty_cache()

            if(key%100==0):
                print(f"Iter No. : {key}/{n_iter} , loss: {round(loss.item(),2)} ")



        for key,data in enumerate(validationloader,0):
            X,y=data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X,batch_norm)
            loss = loss_fn(outputs,y)

            del X
            del y
            del outputs
            torch.cuda.empty_cache()

        epoch+=1
    val_loss.append(loss.item())
    train_loss.append(loss.item())
    # Log metrics to W&B
    wandb.log({
        "Epoch":epoch,
        "Train loss":train_loss[epoch],
        "validation loss":val_loss[epoch],
        "Validation Acc":accuracy(validationloader,model,batch_norm),
    })

    print("Training_accuracy:%.2f" % (accuracy(trainloader,model,batch_norm)))
    print("Validation_accuracy:%.2f" % (accuracy(validationloader,model,batch_norm)))

# fit(batch_norm=1,k=128,file_org=1,F=3,max_epochs=20,batch_size=128,d=0.0,act_fun=nn.ReLU())
def train():
    """
        Train the model with hyperparameters specified in the W&B config.

        Returns:
            None
    """
    wandb.init(config=hyperparameter_defaults)
    config=wandb.config
    fit(batch_norm = config.batch_norm,
        k = config.no_of_layers,
        file_org = config.file_org,
        F = config.Filter_size,
        max_epochs = config.max_epochs,
        batch_size = config.batch_size,
        d = config.drop_out,
        act_fun=nn.ReLU())
   #change
train()