If there are any possible improvements to be made, please make an issue! 

# cifar-resnet-95
This code is a code that achieves a ~95% accuracy on the Cifar-10 dataset using Resnet-18. \
this is very close to, if not the limit for the Resnet architecture. \

Keep in mind, there's still room for a lot of improvement, and a different model architecture will give different results.\
This is mainly a sample of how to optimize trianing for a model, and I wrote this as a way for me to learn different combinations of training methods. \
The Resnet architecture was taken from an already implemented source.

# Side notes
While pytorch lightning speeds up the learning process, the accuracy decreases to about 93% when used.\
For maximum precision, please turn off pytorch lightning.

### Reference to the Cifar-10 dataset
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
