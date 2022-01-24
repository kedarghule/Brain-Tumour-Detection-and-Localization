# Brain-Tumour-Detection-and-Localization

## Problem Statement

We are given almost 4000 MRI images and corresponding mask image which shows the presence of a brain tumour. The problem statement is to detect and localize brain tumours from MRI scans.

## Dataset

The dataset used for this project can be found at this [link](https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation). Due to the size of the dataset, I have no uploaded it on this repository.

## Exploratory Data Analysis

- We start off by finding out the number of items in each class.

![No_Class](https://user-images.githubusercontent.com/41315903/150867868-6d954009-df8c-4189-9099-6a8a10f65d60.png)

- Next, we gain insights into what our image data is plot the MRI scan image and the mask image side by side.
![MRI and mask](https://user-images.githubusercontent.com/41315903/150868087-8d9970c9-a2fa-4f4f-9cc1-32f8a058f5e3.png)

- To gain further insight, we plot the MRI scan images from only sick patients followed by the corresponding mask and then both MRI image and the corresponding mask (in red color) on top of each other.
![MRI mask and overlap](https://user-images.githubusercontent.com/41315903/150868423-9a6b25f3-e504-496c-bba3-995164fc19ee.png)

## Training The Classifier Model to Detect Brain Tumours

- **Dropping Columns Which Are Not Necessary** - We drop the `patient_id` column.
- **Splitting The Data** - We split the dataset into training and testing using scikit-learn.
- **Image Data Generator** - This is used to generate all the images. At the same time, we also rescale or normalize the images. We also make a validation split of 15% from the training data. We use one Image Data Generator for the training and validation dataset and one Image Data Generator for the test dataset.
- **ResNet-50** - We use the ResNet50 architecture pre-trained with the weights of ImageNet dataset. I will be specifying my own dense layer at the end.
- **Classification Head** - We add an Average Pooling 2D layer followed by flattening all the feature maps. After that we have a 256 neuron dense layer with ReLU as the activation function. We apply a 30% dropout followed by another Dense Layer and another dropout. At the output, we have 2 neurons which will indicate if tumour exists or not and the activation function used is 'softmax'.
- **Compiling Our Model** - Loss is specified as categorical_crossentropy, Optimizer is specified to be the Adam optimizer and we use accuracy as our metric.
- **Early Stopping** - We use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience).

## Assessing Our Model That Detects Brain Tumours

- Accuracy: 98.4%
- Confusion matrix is shown below:
- 
![confusionmat](https://user-images.githubusercontent.com/41315903/150873182-a214d1a7-ee85-4e83-a7f2-2d206f7bec7c.png)
- Classification Report is shown below:

![cr](https://user-images.githubusercontent.com/41315903/150873280-961d2ba9-1baf-4b4e-93a0-2e163ddea89e.png)

