import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_model_instance_segmentation(num_classes):
    # Load an instance segmentation model pre-trained on COCO
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
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

    model = get_model_instance_segmentation(num_classes)
    inspect_fpn_output_shapes(model, dummy_img)
