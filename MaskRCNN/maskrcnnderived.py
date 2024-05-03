from instanceSegmentationDataset import PennFudanDataset
import torch
import utils
import os, sys
import json
import torch, torchvision
from torchvision.io import read_image
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
from torchvision.io import read_image
from model import get_model_instance_segmentation
import torchvision.transforms as transforms


from torchvision.transforms import v2 as T

from predict import start_prediction

from seeInputs import show_img

from PIL import Image


current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming 'VisionSystem' is a directory at the same level as this script
# And you want to add it to sys.path
vision_system_path = os.path.join(current_script_dir, '..')

# Add it to sys.path
sys.path.insert(0, vision_system_path)

from visionsystembase import VisionSystemBase

class MaskRCNNDerived(VisionSystemBase):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # num_classes = 2
    config = {}


    def get_transform(self,train):
        transforms = []
        if train:
            transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        return T.Compose(transforms)

    def getPathToModel(self):
        return "./outputs/model.pth"

    def getOutputPath(self):
        return "./outputs/output_img1.png"
    
    def load_model(self, num_classes, model_path, device):
        # Load an instance segmentation model pre-trained on a custom dataset
        model = get_model_instance_segmentation(num_classes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        return model

    
    def load_data(self):
        data_dir = self.config['data_root_dir']
        data_img_dir = self.config['data_img_dir']
        data_mask_dir = self.config['data_mask_dir']
        dataset = PennFudanDataset(data_dir, data_img_dir, data_mask_dir, self.get_transform(train=True))
        dataset_test = PennFudanDataset(data_dir, data_img_dir, data_mask_dir, self.get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=4,
            collate_fn=utils.collate_fn
        )

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=utils.collate_fn
        )
        return {'data_loader' : data_loader, 'data_loader_test': data_loader_test}

    
    def train(self, dataloaders):
        num_classes = self.config['num_classes']
        model = get_model_instance_segmentation(num_classes)

        # move model to the right device
        model.to(self.device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005
        )

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=3,
            gamma=0.1
        )

        num_epochs = 2

        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, dataloaders['data_loader'], self.device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, dataloaders['data_loader_test'], device=self.device)
        
        torch.save(model.state_dict(), self.config['model_save_path'])

    
    def test(self, test_image_path):
        model = self.load_model(self.config['num_classes'], self.config['model_save_path'], self.device)
        start_prediction(test_image_path, model, self.config['output_save_path'])

    
    def visualize_input(self):
        image = read_image(self.config['sample_input_image'])
        mask = read_image(self.config['sample_input_mask'])
        show_img(image, mask)

    
    def visualize_output(self):
        # start_prediction()
        image = read_image(self.config['output_save_path'])
        show_img(image)
        pass

    def get_model_features(self):
        features = []
        def hook(module, input, output):
            features.append(output)
        return features, hook


    
    def visualize_features(self):
        # Instantiate the model
        model = get_model_instance_segmentation(self.config['num_classes'])

        # Initialize a list to hold the features
        features = []

        # Create a hook function
        def hook_fn(module, input, output):
            # This will capture the output of the layer
            features.append(output)

        # Assuming `model.backbone.body` is the attribute where the ResNet layers are stored
        # and you want to attach the hook to the last layer of the ResNet base
        layer = list(model.backbone.body.children())[-1]

        # Register the hook
        handle = layer.register_forward_hook(hook_fn)

        image = Image.open(self.config['sample_input_image'])
        # Convert to tensor and normalize to [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(self.device)  # Move to the same device as model


        # Perform a forward pass (this will trigger the hook)
        model.eval()
        with torch.no_grad():
            _ = model(image_tensor)
        feature_map = features[0][0, 0].cpu().numpy()  # Adjust indexing based on specific use case

        mask = Image.open(self.config['sample_input_mask'])
        output_img = Image.open(self.config['output_save_path'])

        plt.figure(figsize=(16, 8))
        plt.subplot(141)
        plt.title("Image")
        # plt.imshow(image.permute(1, 2, 0))
        plt.imshow(image)

        # plt.figure(figsize=(16, 8))
        plt.subplot(142)
        plt.title("Mask")
        # plt.imshow(image.permute(1, 2, 0))
        plt.imshow(mask)

        # plt.figure(figsize=(16, 8))
        plt.subplot(143)
        plt.title("Output")
        # plt.imshow(image.permute(1, 2, 0))
        plt.imshow(output_img)
    
        plt.subplot(144)
        plt.title("Feature map")
        plt.imshow(feature_map, cmap='viridis')
        plt.colorbar()


        plt.show()


        pass
    

def main():

    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    obj = MaskRCNNDerived()
    obj.config = config
    dl = obj.load_data()
    obj.train(dl)
    # obj.visualize_input()
    # obj.test(config['test_image'])
    # obj.visualize_output()
    # obj.visualize_features()
    

if __name__ == "__main__":
    main()