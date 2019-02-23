# Six Points about effective transfer learning

I would like to share the 6 points that I paied attention when I made my transfer learning model based on MobileNet to classify the image into 5 classes. The MobileNet is one of the lightweight and fast deep learning models to run on the device that has poor computer resources like smart phone.

1. Pick out the still frame from the movie as training dataset
1. Consider the purpose of the model when you conigure your own output layers
1. Set the trainable flag of the batch normarization layers to true
1. Add training dataset more by using data augmentation feature of Keras
1. The model saved at the last epoch is not always the best
1. Convert your model to CoreML model to use it on iOS device

I tried ResNet50 and VGG16 but MobileNet showed similar performance to other two models even though it has very small foot print (15MB). The file size of other two models are more than 100MB, and inference of ResNet50 is 2 times, VGG16 is 4 times slower than MobileNet.

I built highly accurate model in very short term by using high performace model as a base. I trained the model with the images that I picked out from movies. I considered to be able to run the training process from creating data through building my model.

You can easily build your own computer vision model based on MobileNet just only you place the movie files for input and class file for output. Training takes around 60 second per epoch when I run this code on my Mac without GPU.

I hope this jupyter notebook helps when you build your own computer vision model on the edge device as the inference on the edge device will be getting hotter and hotter from now on.
