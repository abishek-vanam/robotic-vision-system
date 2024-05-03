import torch
import timm
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN

def create_convnext_backbone(pretrained=True, trainable_layers=3):
    backbone = timm.create_model('convnext_base', features_only=True, pretrained=pretrained)
    
    print("Feature Indices:", [i for i in range(len(backbone.feature_info.channels()))])
    print("Feature Channels:", backbone.feature_info.channels())
    
    feature_map_indices = [0, 1, 2, 3]  # Example indices, adjust based on actual feature output
    feature_channels = [backbone.feature_info.channels()[i] for i in feature_map_indices]
    
    # Wrap the backbone with an FPN
    # return_layers = {f'{i}': str(idx) for idx, i in enumerate(feature_map_indices)}
    return_layers = {
        'stages_0': '0',
        'stages_1': '1',
        'stages_2': '2'
    }

    backbone = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=feature_channels, out_channels=256)
    return backbone

def get_model_instance_segmentation_convnext(num_classes):
    # Create the ConvNext backbone with FPN
    backbone = create_convnext_backbone()

    # Create the Mask R-CNN model using the ConvNext backbone
    model = MaskRCNN(backbone, num_classes=num_classes)

    # Replace the pre-trained box predictor with a new one
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the pre-trained mask predictor with a new one
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,  # hidden layer size
        num_classes
    )

    return model

def inspect_fpn_output_shapes(model, input_tensor):
    # Forward pass through the backbone
    with torch.no_grad():
        features = model.backbone(input_tensor)
        print({name: feature.shape for name, feature in features.items()})

# Example usage
if __name__ == "__main__":
    num_classes = 10
    dummy_img = torch.rand(1, 3, 224, 224)  # Example input tensor

    model = get_model_instance_segmentation_convnext(num_classes)
    inspect_fpn_output_shapes(model, dummy_img)
