import torch
import torchvision
# from torchvision.models.detection.backbone_utils import backbone_with_fpn
from torchvision.models.detection.backbone_utils import BackboneWithFPN#backbone_with_fpn
import timm
from torch import nn

def create_convnext_backbone():
    # Load ConvNext backbone
    backbone = timm.create_model('convnext_tiny', pretrained=True, features_only=True, out_indices=(1, 2, 3))
    # Number of output channels from the last block
    out_channels = backbone.feature_info.channels()[-1]
    # Creating FPN over the backbone
    backbone = BackboneWithFPN(backbone, return_layers=[0, 1, 2], out_channels=out_channels)
    return backbone

def create_resnet_backbone():
    # Load a pretrained ResNet50 model and remove the fully connected layer
    resnet_backbone = torchvision.models.resnet50(pretrained=True)
    resnet_backbone = nn.Sequential(*list(resnet_backbone.children())[:-2])
    # Number of output channels from the last block
    out_channels = 2048
    # Creating FPN over the backbone
    backbone = BackboneWithFPN(resnet_backbone, return_layers=[2, 3, 4], out_channels=out_channels)
    return backbone

def forward_and_print_shapes(backbone, input_tensor):
    # Forward pass through the backbone
    with torch.no_grad():
        features = backbone(input_tensor)
    # Print shapes of the output features
    print({name: feature.shape for name, feature in features.items()})

# Test the setup
dummy_img = torch.rand(1, 3, 256, 256)  # Batch size, Channels, Height, Width
convnext_backbone = create_convnext_backbone()
resnet_backbone = create_resnet_backbone()

print("ConvNext Output Shapes:")
forward_and_print_shapes(convnext_backbone, dummy_img)

print("\nResNet Output Shapes:")
forward_and_print_shapes(resnet_backbone, dummy_img)
