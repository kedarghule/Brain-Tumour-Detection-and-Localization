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

![confusionmat](https://user-images.githubusercontent.com/41315903/150873182-a214d1a7-ee85-4e83-a7f2-2d206f7bec7c.png)

- Classification Report is shown below:

![cr](https://user-images.githubusercontent.com/41315903/150873280-961d2ba9-1baf-4b4e-93a0-2e163ddea89e.png)

## Building an Image Segmentation Model to Localize Tumours

Now that we have a model with a good accuracy to detect brain tumours from MRI scans, we can move to making an image segmentation model to localize brain tumours on a pixel-level.

- **Data Preprocessing** - We only select that data in which the images which have a mask=1; meaning that the brain tumour exists. This will be our new dataframe.
- **Splitting the Data** - We split the dataset into training, validation and testing set using scikit-learn.
- 
**Next we define two important functions:**
- **Upsampling** - We take the input and use UpSampling2D()
- **The Res-block** - We make a main path (Apply a series of convolutions, batch normalization and use ReLU activation function. We again have a series of convolutions and batch normalization). We also define a short path which only specifies one convolution 2d and batch normalization. Next, we add the short path and the main path, apply the ReLU activation function to it.

Now, we are ready to make our model. We use the **ResUNet model** to localize the brain tumours at pixel-level.
- Stage 1: We have Convolution followed by Batch Normalization followed by another Convolution 2D and followed by another Batch Normalization. After that, we perform max-pooling 2 x 2.
- Stage 2, 3 and 4: We have the Res-block followed by max-pooling 2 x 2.
- Stage 5: Bottleneck. Here we have only one Res-Block.
- Upscale Stage 1, 2, 3 and 4: Perform upsampling followed by a Res-block.
- Final Convolution: This is done to ensure that the final output mask will have the same size as input image.

**Loss Function** - To train the above ResUNet, we use Focal Tversky loss function. We compile the model and we use the Adam optimizer, Tversky function as our metric.

We now train this ResUNet segmentation model to predict the mask.

## Assessing The ResUNet Model

Below is a plot that shows the original MRI scan, the original mask, the predicted mask from our ResUNet model, the original MRI scan with the original mask superimposed on it, the MRI scan with the AI predicted mask superimposed on it.

![segmentation_model](https://user-images.githubusercontent.com/41315903/151061027-c4ac20e7-e6a9-4fc6-820f-91fa90047b42.png)
