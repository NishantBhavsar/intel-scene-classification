![Competition banner](images/banner.png)

[Competition Link](https://datahack.analyticsvidhya.com/contest/practice-problem-intel-scene-classification-challe/)

#### Problem Statement
How do we, humans, recognize a forest as a forest or a mountain as a mountain? We are very good at categorizing scenes based on the semantic representation and object affinity, but we know very little about the processing and encoding of natural scene categories in the human brain. In this problem, you are provided with a dataset of ~25k images from a wide range of natural scenes from all around the world. Your task is to identify which kind of scene can the image be categorized into.

#### Dataset Description

There are 17034 images in train and 7301 images in test data. The categories of scenes and their corresponding labels in the dataset are as follows -
```
'buildings' -> 0
'forest' -> 1
'glacier' -> 2
'mountain' -> 3
'sea' -> 4
'street' -> 5
```
- There are three files provided to you, viz train.zip, test.csv and sample_submission.csv which have the following structure.

| Variable	| Definition |
| ------------- | ----------------- |
| image_name	| Name of the image in the dataset (ID column) |
| label | Category of natural scene (target column) |
 

- train.zip contains the images corresponding to both train and test set along with the true labels for train set images in train.csv

#### Evaluation Metric
The Evaluation metric is accuracy.


### Models
Following models are used :

- ResNet 50 pretrained on ImageNet
- ResNet 101 pretrained on ImageNet
- SE-ResNeXt 101 pretrained on ImageNet
- ResNet 50 pretrained on [CSAILVision places365](https://github.com/CSAILVision/places365) scene classification dataset.

I have used Fast.ai library, it provides easy to use new cutting egde techinques like cyclic learning rate, learning rate finder, etc.

Cyclic learning rate helps to achieve really good score in less number of epochs.


There is a fundamental difference between object classification and scene classification. In object detection our model tries to find an object, so if we look at Class Activation Mapping (CAM) on this model then we can see that it focuses on one point (perticular portion of an image where the object is). Where else in scene classification, scene is covering the entire image, this model takes into consideration the entire scene and we can see that in CAM of Places 365 pre trained model.

The problem with imagenet pretrained models is that they are trained on object classification dataset, so it becomes tough to fine tune (train all layers) the model and get improved result than what we get from just training last layers of the model in scene classification task, it might take lots of epoch to get better result. That's why ResNet 50 pretrained on Places365 works really well in this task, and this model gives  best validation accuracy.


### Image Augmentation

Used these augmentation
- Random cutout
- Rotation
- Horizontal flip 
- Brighness
- Pixle Jitter

Image example with augmentation

![Data sample](images/img_with_aug.png)

### Result

|model|Val Acc| Val TTA Acc |  Info |
|-----|-------|-------------|-------|
| ResNet 50 places | 0.9533 | 0.9506 | first trained on img size 75, then trained on img size 150 |
| ResNet 50 | 0.9463 | 0.9445 | first trained on img size 75, then trained on img size 150 |
| ResNet 101 | 0.9472 | 0.9454 | trained on img size 150 |
| SE ResNext 101 | 0.9436 | 0.9507 | trained on img size 150 |
| Ensemble | - | 0.9554 | average of probabilities of all model for each class |

- I have used Test time augmentation for final submission.
- Took average of probabilities of all model for each class.

- private LB Accuracy score - 0.9544

#### Ensemble model
![ens cm](images/ensemble_cm.png)

#### ResNet 50 Imagenet

![rn50 cm](images/rn50_cm.png)

Class activation mapping (CAM), top losses
![rn50 cam loss](images/rn50_cam_loss.png)

#### ResNet 50 Places 365

![rn50 places cm](images/rn50_plac_part2_cm.png)

Class activation mapping (CAM)
![rn50 places cam ](images/rn50_plac_cam.png)

#### ResNet 101 Imagenet

![rn101 cm](images/rn101_cm.png)

Class activation mapping (CAM), top losses
![rn101 cam ](images/rn101_cam_loss.png)


#### SE ResNeXt 101 Imagenet

![se rnxt 101 cm](images/se_rn101_cm.png)

Class activation mapping (CAM)
![se rnxt 101 cam ](images/se_rn101_cam.png)








