# DL_Assignment2

Convolution Neural Network (CNN)

## PART A

#### CNN Training Script

This script uses PyTorch to train a type of neural network called Convolutional Neural Network (CNN) on a dataset. You can adjust different parts of how the CNN works and how it learns by changing some settings in the script. This includes things like the structure of the CNN, how long it trains for, and certain numbers that affect its performance.

### Usage

1-Get your dataset ready and make sure to tell the script where to find it by updating the data_path setting.

2-Run the training script with the settings you want to use.

### CNN Architecture

The CNN structure is set up in the script named 'DL_ASSIGNMENT_2_PartA.ipynb'. It's made up of several layers that perform specific tasks like detecting patterns in images. These layers include convolutional layers, which are like filters that scan the image, and max-pooling layers, which simplify the information. There's also a fully connected layer that helps make sense of the patterns detected. You can change how the CNN is set up by adjusting different settings in the script.

# Question 1 : DL_Assignment2_PartA(Q1)
let's run each section of the code individually. In the last section, we'll change some parameters and then run the fit function.

# Question 2 : DL_Assignment2_PartA(Q2)
We'll skip the Data Augmentation section to avoid long training times, but the code will still work fine.After that, we'll move on to the section with wandb code and sweep functions. You can log in to wandb and update the authentication IDs as needed for it to run. All the outputs will be available in the output sections.

# Question 3:
written observation based on my plots

# Question 4 : DL_Assignment2_PartA(Q4)
In this part, we've organized the code into different sections. Just run each section one by one. However, please be aware that running the Data Augmentation section will increase the number of data points by four times, which means it will take much longer to run due to the increased dataset size.
The `plot_grid` function plots a 10 x 3 grid of images with their true labels and predicted labels. It can be used to visualize the model's predictions on a subset of the validation dataset.

### References

1. [Weights & Biases Documentation](https://docs.wandb.ai/)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
