import torch
import os, sys
from torchvision.io import read_image
import datahandler
from pathlib import Path
from model import createDeepLabv3
from trainer import train_model
from sklearn.metrics import f1_score, roc_auc_score
import cv2
import matplotlib.pyplot as plt
from predictimg import start_prediction
import json

from torchvision.transforms import v2 as T

current_script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming 'VisionSystem' is a directory at the same level as this script
# And you want to add it to sys.path
vision_system_path = os.path.join(current_script_dir, '..')

# Add it to sys.path
sys.path.insert(0, vision_system_path)

from visionsystembase import VisionSystemBase



class DeepLabV3Derived(VisionSystemBase):
    model = ''
    model_backbone = ''
    learning_rate = ''
    optimizer = ''
    device = ''
    num_classes = 1
    batch_size = 4
    config = {}

    def load_data(self):
        data_dir = self.config['data_root_dir']
        exp_dir = self.config['exp_dir']
        # exp_dir = 'CFExp1'
        data_directory = Path(data_dir)
        exp_directory = Path(exp_dir)
        # Create the dataloader
        img_dir = self.config['data_img_dir']
        mask_dir = self.config['data_mask_dir']

        dataloaders = datahandler.get_dataloader_single_folder(
            data_directory,img_dir, mask_dir,  batch_size=self.batch_size)
        
        if not exp_directory.exists():
            exp_directory.mkdir()
        return dataloaders

    def train(self, dataloaders):
        epochs = 25

        # exp_directory = Path('CFExp1')
        exp_directory = self.config['exp_dir']
        
        model = createDeepLabv3(self.num_classes)
        model.train()
        

        # Specify the loss function
        criterion = torch.nn.MSELoss(reduction='mean')
        # Specify the optimizer with a lower learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Specify the evaluation metrics
        metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

        
        _ = train_model(model,
                        criterion,
                        dataloaders,
                        optimizer,
                        bpath=exp_directory,
                        metrics=metrics,
                        num_epochs=epochs)

        # Save the trained model
        # torch.save(model, exp_directory / self.config['model_save_path'])
        torch.save(model, self.config['model_save_path'])
        pass

    def test(self):
        start_prediction()
        pass

    def visualize_input(self):
        # ino = 2
        # Read  a sample image and mask from the data-set
        # img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)
        # mask = cv2.imread(f'./CrackForest/Masks/{ino:03d}_label.PNG')
        
        img = cv2.imread(self.config['sample_input_image']).transpose(2,0,1).reshape(1,3,320,480)
        mask = cv2.imread(self.config['sample_input_mask'])

        plt.figure(figsize=(10,10))
        plt.subplot(131)
        plt.imshow(img[0,...].transpose(1,2,0))
        plt.title('Image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(mask)
        plt.title('Ground Truth')
        plt.axis('off')
        plt.show()


    def visualize_output(self):
        start_prediction()
        pass

    def visualize_features(self):
        pass
    pass

# "python main.py --data-directory CrackForest --exp_directory CFExp"

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    

    obj = DeepLabV3Derived()
    obj.config = config
    dl = obj.load_data()
    obj.train(dl)
    # obj.visualize_output()
    pass

if __name__ == "__main__":
    main()