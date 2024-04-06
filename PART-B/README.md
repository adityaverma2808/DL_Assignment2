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

### Arguments

The script accepts the following arguments:

| Argument                    | Description                                                                                                      |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| -in / --input_channel       | Number of input channels for the CNN. Default is 3.                                                              |
| -op / --output_neuron       | Number of output neurons for the CNN. Default is 10.                                                             |
| -fo / --filter_organisation | Filter organization strategy for the CNN. Options: 'same', 'double', 'half', 'alt', 'd_alt'. Default is 'same'.  |
| -nf / --no_of_filters       | Number of filters for each convolutional layer. Default is 64.                                                   |
| -fs / --filter_size         | Size of filters for each convolutional layer. Default is [3, 3, 3, 3, 3].                                        |
| -st / --stride              | Stride size for all convolutional layers. Default is 1.                                                          |
| -pd / --padding             | Padding size for all convolutional layers. Default is 1.                                                         |
| -pfs / --pool_filter_size   | Pooling filter size for all layers. Default is 1.                                                                |
| -pst / --pool_stride        | Pooling stride size for all layers. Default is 2.                                                                |
| -ppd / --pool_padding       | Pooling padding sizes for all layers. Default is 0.                                                              |
| -act / --activation         | Activation function for the CNN. Options: 'ReLU', 'GELU', 'SiLU', 'Mish', 'ELU', 'LeakyReLU'. Default is 'ReLU'. |
| -da / --data_augmentation   | Whether to apply data augmentation. Options: 'Yes', 'No'. Default is 'No'.                                       |
| -bn / --batch_normalization | Whether to use batch normalization. Options: 'Yes', 'No'. Default is 'Yes'.                                      |
| -do / --dropout             | Dropout probability. Default is 0.3.                                                                             |
| -dls / --dense_size         | Number of neurons in the dense layer. Default is 256.                                                            |
| -bs / --batch_size          | Batch size for training. Default is 32.                                                                          |
| -is / --img_size            | Image size for resizing. Default is 224.                                                                         |
| -ep / --epochs              | Number of epochs for training. Default is 10.                                                                    |
| -lr / --learning_rate       | Learning rate for optimization. Default is 0.0001.                                                               |
| -ftp / --flag_total_params  | Flag for printing total parameters of the model. Options: 1 (Yes), 0 (No). Default is 0.                         |
| -opt / --optimizer          | Optimizer choice for training. Options: 'adam', 'sgd'. Default is 'adam'.                                        |
| -dp / --data_path           | Path to the dataset.                                                                                             |
| -p / --plot                 | Whether to plot a grid of images with predictions. Options: 1 (Yes), 0 (No). Default is 0.                       |
| -wl / --wandb_log           | Whether to log training metrics on wandb. Options: 1 (Yes), 0 (No). Default is 1.                                |
| -wp / --wandb_project       | Project name used to track experiments in Weights & Biases dashboard.                                            |
| -we / --wandb_entity        | Wandb Entity used to track experiments in the Weights & Biases dashboard.                                        |


### References

1. [Weights & Biases Documentation](https://docs.wandb.ai/)
2. [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
