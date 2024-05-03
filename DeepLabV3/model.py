""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(num_classes=1, back_bone = 'resnet101'):
    """DeepLabv3 class with custom head

    Args:
        num_classes (int, optional): The number of classes
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    if back_bone == 'resnet101':
        model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    elif back_bone == 'resnet50':
        model = models.segmentation.deeplabv3_resnet50(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, num_classes)
    # Set the model in training mode
    model.train()
    return model
