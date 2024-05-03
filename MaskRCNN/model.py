import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # breakpoint()
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model



# import torch
# import torchvision
# from torchvision.models.detection import MaskRCNN
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
# from torchvision.models.detection.backbone_utils import BackboneWithFPN#backbone_with_fpn
# import timm

# def get_convnext_backbone1(pretrained=True, trainable_layers=3):
#     backbone = timm.create_model('convnext_base', features_only=True, pretrained=pretrained)
    
#     print("Feature Indices:", [i for i in range(len(backbone.feature_info.channels()))])
#     print("Feature Channels:", backbone.feature_info.channels())
    
#     feature_map_indices = [0, 1, 2, 3]  # Example indices, adjust based on actual feature output
#     feature_channels = [backbone.feature_info.channels()[i] for i in feature_map_indices]
    
#     # Wrap the backbone with an FPN
#     # return_layers = {f'{i}': str(idx) for idx, i in enumerate(feature_map_indices)}
#     return_layers = {
#         'stages_0': '0',
#         'stages_1': '1',
#         'stages_2': '2'
#     }

#     backbone = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=feature_channels, out_channels=256)
#     return backbone

# import timm
# from torchvision.models.detection.backbone_utils import BackboneWithFPN

# def get_convnext_backbone(pretrained=True, trainable_layers=3):
#     # Create a ConvNeXt model from timm with features_only=True to get feature pyramid outputs
#     backbone = timm.create_model('convnext_base', features_only=True, pretrained=pretrained)
    
#     # Define feature map indices based on the actual feature outputs from the model
#     feature_map_indices = [1, 2, 3]  # We skip the first layer (index 0 with 128 channels)
#     feature_channels = [256, 512, 1024]  # Corresponding to indices 1, 2, 3

#     # Map the correct stages to the corresponding indices for FPN
#     return_layers = {f'stages_{i}': str(idx) for idx, i in enumerate(feature_map_indices)}

#     # Wrap the backbone with an FPN
#     #1. Forward with your ConvNExt - check the shape
#     #2. Forward with Renset + FPN - check the shape
#     #3. Find the difference
#     # https://pytorch.org/vision/main/generated/torchvision.ops.FeaturePyramidNetwork.html
#     # https://pytorch.org/vision/main/generated/torchvision.ops.FeaturePyramidNetwork.html
    
#     backbone = BackboneWithFPN(backbone, return_layers=return_layers, in_channels_list=feature_channels, out_channels=1024)
#     return backbone


# def get_model_instance_segmentation(num_classes):
#     # Replace the ResNet backbone with ConvNeXt from timm
#     backbone = get_convnext_backbone()
#     model = MaskRCNN(backbone, num_classes=num_classes)

#     # Get number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     # Replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # Now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256
#     # Replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(
#         in_features_mask,
#         hidden_layer,
#         num_classes
#     )

#     return model
