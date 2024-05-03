import matplotlib.pyplot as plt
from torchvision.io import read_image

def show_img(img = '', mask = ''):
    # image = read_image("data/PennFudanPed/PNGImages/FudanPed00040.png")
    # mask = read_image("data/PennFudanPed/PedMasks/FudanPed00040_mask.png")
    # img = read_image('outputs/output_img.png')

    
    plt.figure(figsize=(16, 8))
    if img != '':
        print("test")
        plt.subplot(121)
        plt.title("Image")
        plt.imshow(img.permute(1, 2, 0))
    if mask != '':
        plt.subplot(122)
        plt.title("Mask")
        plt.imshow(mask.permute(1, 2, 0))
    plt.show()

# show_img()