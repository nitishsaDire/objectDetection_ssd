# SSD
This is a PyTorch implemented model for object detection.

The aim is to build a model that can detect and localize objects (belonging to a set of categories) present in the image.

# Model Details
There are two models that I have implemented, one is ResNet-34 based, and the other is VGG based (as mentioned in
the SSD research paper).

Results on a smaller training-validation dataset showed that VGG based giving better results, so here I will discuss
that only. VGG backbone is used, with the fc layers converted to convolutional layers
by using sub-sampling techniques.


Features from different depths are all used in the task because these features
have different scales as it helps in detecting and localizing the objects of
various sizes. As mentioned in the paper, "To handle different object scales,
some methods [4,9] suggest processing the image at different sizes and combining the
results afterward. However, by utilizing feature maps from several different layers in a
single network for prediction we can mimic the same effect, while also sharing parameters across all object scales."

# Loss Function
There is a fixed set of priors of different scales and aspect ratios pre-defined for all
input images. The priors which have IOU with ground truth boxes more than some threshold
are called are positive one, otherwise negative one. For the loss function
CCE loss is used for classification and L1 loss is used for generating bounding box
coordinates.

Hard Negative Mining is used because most of the priors don't have an object, which can lead to a model which is trained to detect background, rather than
An object. To balance this problem what can we do is we can find out the anchors where the model was sure it is no background, but ground-truth was background,
i.e those anchors where the model detects background poorly and we can include those anchors' cce in the loss.
The combined effect is model is trained to find objects and also trained to differentiate whether background or not.


# Results
I have trained on PASCAL VOC 2007 + 2012 trainval dataset, 

Loss function is plotted below:




