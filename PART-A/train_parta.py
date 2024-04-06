# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-wp","--wandb_project",help="Project name used to track experiments in Weights & Biases dashboard",default="DL_Assignment_2")
parser.add_argument("-we","--wandb_entity",help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default="cs23m008")
parser.add_argument("-dp","--dataset_path",help="Number of epochs to train neural network.",default="/content/drive/MyDrive/dl-assigment-2/inaturalist_12K/train")
parser.add_argument("-e","--epochs",help="Number of epochs to train neural network.",choices=['10','20','30'],default=10)
parser.add_argument("-nf","--num_filters",help="choices: ['16', '32']",choices=['16', '32'],default=32)
parser.add_argument("-sz","--neurons_dense",help=f"choices: ['32', '128', '512', '1024']",choices=['32', '128', '512', '1024'],default=512)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
hyperparameter_defaults= {
             "batch_norm":0, # Whether to use batch normalization (0 for False, 1 for True)
             "no_of_layers":16, # Number of convolutional layers
             "file_org":1, # Factor by which filter size increases
             "Filter_size":3, # Size of the filters
              "max_epochs":2, # Maximum number of epochs for training
              "batch_size":128, # Batch size for training
              "drop_out":0.3 # Dropout probability
}
dataset_url = "/content/drive/MyDrive/dl-assigment-2/inaturalist_12K/train"
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

class CnnModel(nn.Module):
    def __init__(self,is_batch_norm,filter_size,file_org,F,d,actication_fun):
        """
        Initialize the CNN model.

        Args:
            is_batch_norm (bool): Whether to use batch normalization.
            filter_size (int): Size of the filters.
            file_org (float): Factor by which filter size increases.
            F (int): Filter size for convolutional layers.
            d (int): Pooling stride.
            activation_fun: Activation function to use.
        """
        super(CnnModel, self).__init__()
        if is_batch_norm:
            self.cnn_model = nn.Sequential(
                nn.Conv2d(3,filter_size,F),
                nn.BatchNorm2d(int(filter_size)),
                actication_fun,
                nn.MaxPool2d(2,stride=1),

                nn.Conv2d(filter_size,int(filter_size*file_org),F),
                nn.BatchNorm2d(int(filter_size*(file_org**1))),
                actication_fun,
                nn.MaxPool2d(2,stride=1),

                nn.Conv2d(int(filter_size*file_org),int(filter_size*(file_org**2)),F),
                nn.BatchNorm2d(int(filter_size*(file_org**2))),
                actication_fun,
                nn.MaxPool2d(2,stride=1),

                nn.Conv2d(int(filter_size*(file_org**2)),int(filter_size*(file_org**3)),F),
                nn.BatchNorm2d(int(filter_size*(file_org**3))),
                actication_fun,
                nn.MaxPool2d(2,stride=2),

                nn.Conv2d(int(filter_size*(file_org**3)),int(filter_size*(file_org**4)),int(F)),
                nn.BatchNorm2d(int(filter_size*(file_org**4))),
                actication_fun,
                nn.MaxPool2d(2,stride=2),

            )
        else:
            self.cnn_model = nn.Sequential(
            nn.Conv2d(3,filter_size,F),
            actication_fun,
            nn.MaxPool2d(2,stride=1),

            nn.Conv2d(filter_size,int(filter_size*file_org),F),
            actication_fun,
            nn.MaxPool2d(2,stride=1),

            nn.Conv2d(int(filter_size*file_org),int(filter_size*(file_org**2)),F),
            actication_fun,
            nn.MaxPool2d(2,stride=1),

            nn.Conv2d(int(filter_size*(file_org**2)),int(filter_size*(file_org**3)),F),
            actication_fun,
            nn.MaxPool2d(2,stride=2),

            nn.Conv2d(int(filter_size*(file_org**3)),int(filter_size*(file_org**4)),int(F)),
            actication_fun,
            nn.MaxPool2d(2,stride=2),

            output_dim=x_dim-self.F+1,

            )


        x_dim=32
        y_dim=x_dim-F+1
        # Calculate dimensions after each convolutional layer and max pooling
        x_dim=y_dim-2
        y_dim=x_dim-F+1
        y_dim=y_dim-1

        x_dim=y_dim
        y_dim=x_dim-F+1
        y_dim=y_dim-1

        x_dim=y_dim
        y_dim=x_dim-F+1
        y_dim=int(y_dim/2)

        x_dim=y_dim
        y_dim=x_dim-F+1
        y_dim=y_dim/2
        y_dim=int(y_dim)
        # Define the fully connected layers
        self.fc_model = nn.Sequential(
           nn.Linear(int(y_dim*y_dim*filter_size*(file_org**4)),120),
           nn.ReLU(),
           nn.Dropout(d),
           nn.Linear(120,10)
        )
    def forward(self, x,batch_norm):
        """
        Forward pass of the model.

        Args:
          x (torch.Tensor): Input data.
          batch_norm (bool): Whether to use batch normalization.

        Returns:
          torch.Tensor: Output of the model.
        """
        x = self.cnn_model(x)
        return self.fc_model(x.view(x.size(0), -1))


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
    model = CnnModel(is_batch_norm = batch_norm,
                        filter_size = k,
                        file_org = file_org,
                        F = F,
                        d = d,
                        actication_fun = act_fun).to(device)
    # Define optimizer and loss function
    opt = optim.Adam(model.parameters())
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

def run_sweep():
    wandb.init(project="DL_Assignment_2",
           config={
             "batch_norm":0, # Whether to use batch normalization (0 for False, 1 for True)
             "no_of_layers":16, # Number of convolutional layers
             "file_org":1, # Factor by which filter size increases
             "Filter_size":3, # Size of the filters
              "max_epochs":2, # Maximum number of epochs for training
              "batch_size":128, # Batch size for training
              "act_fun":nn.ReLU(), # Activation function to use
              "drop_out":0.3 # Dropout probability
           })
    
    sweep_config={
        "name":"Assignment_2_PART_A", # Name of the sweep
        "method":"grid", # Method of the sweep (grid search)
        "metric":{
            'name':'accuracy', # Metric to optimize
            'goal' : 'maximize' # Goal of the optimization (maximize accuracy)
        },
        "parameters":{
        "batch_norm":{
            "values":[0] # Values for batch normalization (only one value)
        },
        "no_of_layers":{
            "values":[16,32] # Values for the number of layers
        },
        "file_org":{
            "values":[0.5,1,2]  # Values for the factor by which filter size increases
        },
        "Filter_size":{
            "values":[3] # Values for filter size
        },
        "max_epochs":{
            "values":[20] # Values for maximum number of epochs
        },
        "batch_size":{
            "values":[128]  # Values for batch size
        },
        "drop_out":{
            "values":[0.2,0.5]  # Values for dropout probability
        }


        }
    }
    sweep_id12=wandb.sweep(sweep_config,entity="cs23m008",project="DL_Assignment_2")
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
    wandb.agent(sweep_id12,train)

# Question 3
# run_sweep()

def createTrainDataLoader():
    """
    Create data loaders for training set with various transformations.

    Returns:
        tuple: Tuple containing data loaders for different transformations.
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
    # Define transformations for vertical flipping augmentation
    transform_vertical = transforms.Compose([
        transforms.RandomResizedCrop(op_size),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        ])
    # Define transformations for inverting augmentation
    transform_Invert= transforms.Compose([
        transforms.RandomResizedCrop(op_size),
        transforms.RandomInvert(),
        transforms.ToTensor(),
        ])

    dataset_url = "/content/drive/MyDrive/dl-assigment-2/inaturalist_12K/train"

    # Load the training image data with different transformations
    img_data = torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_train)
    img_data_hori= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_horizontal)
    img_data_vert= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_vertical)
    img_data_inve= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_Invert)
    # Combine all training datasets
    img_data = img_data + img_data_inve + img_data_vert + img_data_hori

    # Split the combined training dataset into train and validation sets
    img_data = train_val_dataset(img_data)

    # Get train and validation subsets for training
    X_train=img_data['train']
    X_Valid=img_data['val']


    # Define the directory containing the test dataset
    dataset_url = "/content/drive/MyDrive/dl-assigment-2/inaturalist_12K/val"
    # Load the test image data with different transformations
    test_img_data = torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_train)
    test_img_data_hori= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_horizontal)
    test_img_data_vert= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_vertical)
    test_img_data_inve= torchvision.datasets.ImageFolder(root= dataset_url,  transform=transform_Invert)
    # Combine all test datasets
    test_img_data = test_img_data + test_img_data_inve + test_img_data_vert + test_img_data_hori


    # Create data loaders for training, validation, and test sets
    trainloader = torch.utils.data.DataLoader(X_train, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_img_data, batch_size=128, shuffle=True)
    validationloader = torch.utils.data.DataLoader(X_Valid, batch_size=128, shuffle=False)

    # Get an iterator for trainloader
    dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # Return trainloader and testloader
    return trainloader,testloader

def train_fit(batch_norm,k,file_org,F,max_epochs,batch_size,d,act_fun=nn.ReLU()):
    """
    Train the model with given hyperparameters.

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
        torch.nn.Module: Trained model.
    """
    # Initialize the model
    model = CnnModel(is_batch_norm = batch_norm,
                        filter_size = k,
                        file_org = file_org,
                        F = F,
                        d = d,
                        actication_fun = act_fun).to(device)
    # Define optimizer and loss function
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    train_loss,test_loss=[],[]

    # Calculate number of iterations per epoch
    n_iter=np.ceil(8999/batch_size)
    # Create data loaders for training and test sets
    trainloader,testloader = createTrainDataLoader()
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



        for key,data in enumerate(testloader,0):
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
    test_loss.append(loss.item())
    train_loss.append(loss.item())
    # Log metrics to W&B
    wandb.log({
        "Epoch":epoch,
        "Train loss":train_loss[epoch],
        "Testing loss":test_loss[epoch],
        "Test Acc":accuracy(testloader,model,batch_norm),
    })

    print("Training_accuracy:%.2f" % (accuracy(trainloader,model,batch_norm)))
    print("Test_accuracy:%.2f" % (accuracy(testloader,model,batch_norm)))
    return model

best_hyperparameter = {
             "batch_norm":0, # Whether to use batch normalization
             "no_of_layers":16, # Number of convolutional layers
             "file_org":1, # Factor by which filter size increases
             "Filter_size":3, # Size of the filters
              "max_epochs":2, # Maximum number of epochs for training
              "batch_size":128, # Batch size for training
              "drop_out":0.3 # Dropout probability
}

def train_with_best_params():
  """
    Train the model with the best hyperparameters.

    Returns:
        torch.nn.Module: Trained model.
  """
  # Initialize W&B with best hyperparameters
  wandb.init(config=best_hyperparameter)
  config=wandb.config
  # Train the model with the best hyperparameters
  return train_fit(batch_norm = config.batch_norm,
      k = config.no_of_layers,
      file_org = config.file_org,
      F = config.Filter_size,
      max_epochs = config.max_epochs,
      batch_size = config.batch_size,
      d = config.drop_out,
      act_fun=nn.ReLU())

def print_grid(best_model):
    transform_test = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Load the testing data
    sample_classes = ('Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia')
    testing_data= torchvision.datasets.ImageFolder(root= "/content/train_local/inaturalist_12K/val",  transform=transform_test)
    testdata_loader = torch.utils.data.DataLoader(testing_data, batch_size=30, shuffle=True)
    testdata_itr = iter(testdata_loader)
    sample_iamges, sample_labels = testdata_itr.next()
    # Perform inference on sample images using the best model
    out_test_c = best_model(sample_iamges,best_hyperparameter.batch_norm)
    imag=sample_iamges[0].unsqueeze(0)
    out_test_c = best_model(imag,best_hyperparameter.batch_norm)
    # Convert tensor to numpy array for visualization
    npimg = np.transpose(sample_iamges[0], (1, 2, 0))

    train_img,img_title = [],[]
    # Iterate through the sample images
    for i in range(30):
        imag=sample_iamges[i].unsqueeze(0)
        out_test_c = best_model(imag,best_hyperparameter.batch_norm) # Apply Forward propogation to already trained model.h
        # Convert tensor image to PIL image for logging to W&B
        train_img.append(transforms.ToPILImage()(torch.squeeze(sample_iamges[i])).convert("RGB"))
        # Get the predicted and correct labels
        predicted_title = sample_classes[out_test_c.argmax(dim=1).item()]
        correct_title = sample_classes[sample_labels[i].item()]
        combine_title = str("Actual Label : ") + correct_title + "\n" + str("Predicted Label : ") + predicted_title
        img_title.append(combine_title)
    # Initialize W&B project
    wandb.init(project="DL_Assignment_2", entity="cs23m008")
    # Log the sample images and their predictions to W&B
    wandb.log({"sample image and prediction from the test dataset": [wandb.Image(img, caption=lbl) for img,lbl in zip(train_img,img_title)]})

# Question 4-a
# Train the model with the best hyperparameters and get the best model
best_model = train_with_best_params()

# Question 4-b

# print_grid(best_model)