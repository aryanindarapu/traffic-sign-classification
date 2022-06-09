# GTSRB Traffic Sign Classification
## Introduction
The goal of this project is to use deep learning models to classify a given traffic sign based on its features. To achieve this, I created several convolutional neural network models using Tensorflow and Keras frameworks. The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which can be found [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## GTSRB Dataset
The dataset consists of 49639 total images, where each image is 32x32 pixels and has 3 channels (RGB). Each pixel is saved using an 8 bit unsigned integer giving a total of 256 possible values. The breakdown of the dataset is as follows:
- Training Set: 34799 images
- Validation Set: 4410 images
- Testing Set: 12630 images

Additionally, each image is classified as one of the 43 unique classes, where each class is a specific type of traffic sign.

## Procedure
There were two approaches taken to create these models.
1. Using the RGB Channels
2. Using the Grayscale Channel
Through each method, 9 different models were tested, with each model differing slightly from the previous model. For example, a model may use the `adam` optimizer, whereas the previous one used the `rmsprop` optimizer. Also, the grayscale base model and continuations are a little differen to fit with the singular channel and to make it the best possible model.

## Results
NOTE: Not all models will be shown; only the most notable models and the best model for the procedure. To see all the models and reasons for the changes, please see the `Model Interation Accuracy.xlsx` file.

### RGB Channels
|     Model Name    | Description                                                                              | Testing Accuracy |
|:-----------------:|------------------------------------------------------------------------------------------|:----------------:|
| base_convnet      | Basic CNN with 3 Conv2D layers and 1 Dense layer.                                        | 85.3%            |
| with_dropout      | Data augmentation and dropout added, along with an extra Dense layer before the dropout. | 95.28%           |
| with_sigmoid      | Sigmoid activation used instead of relu.                                                 | 90.32%           |
| with_padding_conv | Padding added to dropout model with adam optimizer.                                      | 94.11%           |
| with_relu_adam    | Adam optimizer used (instead of rmsprop) with relu activation.                           | 96.41%           |

As seen above, the model using the `relu` activation layers and the `adam` optimizer seem to perform the best on the test set.

### Grayscale Channel
|     Model Name    | Description                                                                              | Testing Accuracy |
|:-----------------:|------------------------------------------------------------------------------------------|:----------------:|
| base_convnet      | Basic CNN with 2 Conv2D layers and 1 Dense layer.                                        | 90.8%            |
| with_dropout      | Data augmentation and dropout added, along with an extra Dense layer before the dropout. | 92.99%           |
| with_sigmoid      | Sigmoid activation used instead of relu.                                                 | 93.85%           |
| with_relu_adam    | Adam optimizer used (instead of rmsprop) with relu activation.                           | 95.65%           |
| with_padding_conv | Padding added to dropout model with adam optimizer.                                      | 96.45%           |

As seen above, the model using the `relu` activation layers, `adam` optimizer, and padding seem to perform the best on the test set.

## Discussion
As seen above, both procedures seem to do very well on the test set, with both achieving at least 96% accuracy. Moreover, since the dataset is so small, data augmentation helped make the model even better with a larger dataset. That being said, there are several differences between the two:
- The RGB channels seemed to reach a better accuracy much quicker, i.e. the inital models achieve at least 93% accuracy. The grayscale, on the other hand, seemed to perform much worse until some of the later models.
- Regardless of the procedure, the `sigmoid` layers seemed to do worse on average, likely due to vanishing gradients. For that reason, the `relu` layers were using.
- Once again, it also seemed that the `adam` optimizer performed much better than the `rmsprop` optimizer, which was expected.
- More epochs usually meant overfitting occurred much more quickly.
- The grayscale channel models, on average, performed about 40% quicker than the RGB channel models. However, the RGB channels seemed to overfit less.

## Conclusion
Although the RGB Model seemed to be better and more accurate for a larger dataset, in a real-world scenario, the grayscale model, which is significantly quicker and requires less data, would likely be used. 

## Testing the Model
For easy use and testing of both the RGB Model and the Grayscale model, I created a GUI using the tkinter package. There are three steps to using this GUI: 
1. As seen in the top left, choose the type of model must be chosen (either "RGB Model" or "Grayscale Model").
2. Upload an image after clicking the "Upload an Image" button. The selected image should pop up in the center of the GUI and a button that says "Classify Image" should appear.
3. Click the "Classify Image" button, and the chosen model will predict the class of the traffic sign and display it above the image.

<img src="/gui_imgs/gui1.png" width="325" />    <img src="/gui_imgs/gui2.png" width="325">    <img src="/gui_imgs/gui3.png" width="325"> 





