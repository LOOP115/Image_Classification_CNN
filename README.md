# Convolutional neural networks for image classification

### [Specification](resources/ass2-spec.pdf)

### [Dataset](resources/yoga32.zip)



## Overview

* Use CNNs to classify a yoga pose dataset, yoga32.
* This dataset, based on a dataset created by Anastasia Marchenkova, includes **590** RGB colour images from **10** classes.
* Images have been downsampled to **32x32** pixels.
* We have provided a **train/test split** with **520 images for training/validation and 70 for testing** – you should use the provided split throughout this assignment.



## Task

### 1.1. CNN implementation

* Implement the CNN architecture shown above in Figure 1. ✔
* Use ReLU activation functions for all layers except the final layer, which should use the Softmax activation function. ✔
* Use the Adam optimiser and SparseCategoricalCrossentropy loss. ✔
* Train this on the yoga32 dataset – what do you observe?

### 1.2. Regularisation and data augmentation

* Modify the basic architecture by adding some form of (a) **regularisation** and (b) **data augmentation**.
* Train your new network on the yoga32 dataset – how does the training performance change?
* Your write-up should include a brief **description and justification** of your choice of regularisation
  and data augmentation schemes.
* It should also show the **plots of training and validation accuracy** for the original network (without regularisation+data augmentation) and the network with these modifications and explain any differences that you observe in the training behaviour.

### 2. Error analysis

* Evaluate your network from part 1.2 on the yoga32 test set.
* In your write-up, present the overall classification accuracy and the average accuracy for each of the 10 classes.
* Explain the performance of the CNN model, using example images from the test set to illustrate your discussion.
* What classes/images were difficult for this model, and why?

### 3. Visualisation

* Visualise the feature space that your network uses to classify images by implementing a nearest neighbour analysis.
* Use the embedding from the last convolutional layer of your network from part 1.2 after it has been maxpooled (e.g., extract this layer at the point at which it is flattened and sent to the classification layer).
* To visualise how images are organised in this feature space, implement a nearest neighbour analysis.
* For each test image, find the 5 nearest neighbours in the training set.
* Use **Euclidean distance** to compare the feature vector from the test image to the feature vectors of the training images.
* In your write-up, show nearest neighbours for multiple test images to illustrate the feature space and explain your model’s performance.
* Critically evaluate your model – has it learned a good feature space for this classification task?



## Submission

* You should make two submissions on the LMS: your code and a short written report explaining your method and results.
* The response to each question should be **no more than 500 words**.
* Please submit your code and written report separately under the **Assignment 2: Code** and the **Assignment 2: Report** links on Canvas.
* Your code submission should include the Jupyter Notebook (please use the provided template) with your code and any additional files we will need to run your code, if any (do not include the yoga32 dataset).
* Your written report should be a .pdf with your answers to each of the questions. The report should address the questions posed in this assignment and include any images, diagrams, or tables required by the question.

