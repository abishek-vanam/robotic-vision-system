import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.io import read_image
import torch
import os
from main import getPathToModel, getOutputPath
from model import get_model_instance_segmentation

from main import get_transform

def predict_img(model, device, image_path, output_path):
    image = read_image(image_path)
    eval_transform = get_transform(train=False)

    model.eval()
    with torch.no_grad():
        x = eval_transform(image)
        x = x[:3, ...].to(device)
        predictions = model([x, ])
        pred = predictions[0]

    image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(torch.uint8)
    image = image[:3, ...]
    pred_labels = [f"pedestrian: {score:.3f}" for label, score in zip(pred["labels"], pred["scores"])]
    pred_boxes = pred["boxes"].long()
    output_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors="red")

    masks = (pred["masks"] > 0.7).squeeze(1)
    output_image = draw_segmentation_masks(output_image, masks, alpha=0.5, colors="blue")

    plt.figure(figsize=(12, 12))
    plt.imshow(output_image.permute(1, 2, 0))
    plt.title("Output")
    plt.savefig(output_path,bbox_inches='tight')
    plt.show()
    plt.close()



# def load_model(num_classes, model_path, device):
#     # Load an instance segmentation model pre-trained on a custom dataset
#     model = get_model_instance_segmentation(num_classes)
#     model.load_state_dict(torch.load(model_path))
#     model.to(device)
#     return model

def start_prediction(image_path, model, output_path):
    # Define the path for the image you want to visualize
    # image_path = "data/PennFudanPed/PNGImages/FudanPed00040.png"

    # Define the output path for the visualization
    # output_path = getOutputPath()

    # Set the device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define the number of classes and the path to the model
    # num_classes = 2  # background and person
    # model_path = getPathToModel()

    # # Load the pre-trained model
    # model = load_model(num_classes, model_path, device)

    # Perform the visualization
    predict_img(model, device, image_path, output_path)

    print("Visualization complete! Check the output at:", output_path)


# start_prediction()