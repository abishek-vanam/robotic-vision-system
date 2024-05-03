import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def start_prediction():
    # Load the trained model 
    model = torch.load('./CFExp1/weights.pt')
    # Set the model to evaluate mode
    model.eval()

    # # Read the log file using pandas into a dataframe
    # df = pd.read_csv('./CFExp1/log.csv')

    # # Plot all the values with respect to the epochs
    # df.plot(x='epoch',figsize=(15,8))

    # print(df[['Train_auroc','Test_auroc']].max())

    ino = 2
    # Read  a sample image and mask from the data-set
    img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)
    mask = cv2.imread(f'./CrackForest/Masks/{ino:03d}_label.PNG')
    with torch.no_grad():
        # a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
        a = model(torch.from_numpy(img).type(torch.FloatTensor)/255)


    # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
    plt.hist(a['out'].data.cpu().numpy().flatten())

    # Plot the input image, ground truth and the predicted output
    plt.figure(figsize=(10,10))
    plt.subplot(131)
    plt.imshow(img[0,...].transpose(1,2,0))
    plt.title('Image')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(mask)
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.2)
    plt.title('Segmentation Output')
    plt.axis('off')
    plt.show()
    plt.savefig('./CFExp1/SegmentationOutput.png',bbox_inches='tight')

# start_prediction()