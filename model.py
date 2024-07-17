"""Define the model architecture in the form of a single class inheriting from `torch.nn.Module`. In this case, we use
the ResNet-18 model from Torchvision.

From GitHub repository pytorch/serve: examples/image_classifier/resnet_18/model.py
"""

from torchvision.models.resnet import ResNet, BasicBlock


class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2, 2, 2, 2])
