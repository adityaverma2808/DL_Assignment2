# DL_Assignment2

Convolution Neural Network (CNN)

## PART B

### Functions Description

#### Training Function (`Train`):

This function trains the model for a certain number of times called epochs. It goes through the dataset, looks at the data, calculates how wrong its guesses are, and then tries to adjust itself to make better guesses next time.

#### Layer Freezing Function (`freeze_layers`):

This function stops certain parts of the model from learning. You can choose to stop the first few layers, some layers in the middle, or the last few layers of the model from learning anything new during training.

#### Data Loading Function (`load_data`):

This function reads the dataset from a folder you specify. It makes sure the images are all the same size, adjusts their colors to be more consistent, and adds some variations to the images to make the model more robust.

#### Fine-Tuning Model Function (`fine_tune_model`):

This main function is in charge of adjusting the GoogleNet model to work better with our specific dataset. It gets the data ready, sets up the model, decides which parts of the model should stay the same and which should change, figures out how to measure how well the model is doing, and then teaches the model how to improve using the training function.

### References

1. [Weights & Biases Documentation](https://docs.wandb.ai/)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
