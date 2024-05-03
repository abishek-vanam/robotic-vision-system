from abc import ABC, abstractmethod

class VisionSystemBase(ABC):

    model = ''
    model_backbone = ''
    learning_rate = ''
    optimizer = ''
    device = ''

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def train(self, dataloaders):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def visualize_input(self):
        pass

    @abstractmethod
    def visualize_output(self):
        pass

    @abstractmethod
    def visualize_features(self):
        # Multiply feature map with the image to see the heatmap of the image to see what region of the image is capture or not
        
        # https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055
        # We can visualise this below:
        # Show heatmap along with the input image for comparison
        
        # Change the backbone using timm API based on the string given in the config file. It has lots of backbones configured
        
        
        # model.backbone = timmm () --> check dimesions
        # Set the model in_channels and backbone using timm.
        # ConvNext, vit, swinTransformer
        pass