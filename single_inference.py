
import torch
import cv2
import types
import numpy as np
from film.medical_net.model import generate_model
import matplotlib.pyplot as plt
import math




class FeatureExtractor():

    def __init__(self):

        opt = types.SimpleNamespace()
        opt.model = 'resnet'
        opt.model_depth = 50
        opt.resnet_shortcut = 'B'
        opt.n_seg_classes = 10
        opt.no_cuda = True
        opt.gpu_id = [0]
        opt.input_W = 512
        opt.input_H = 512
        opt.input_D = 1
        opt.phase = 'features'
        opt.pretrain_path = 'film/medical_net/pretrained/resnet_50_23dataset.pth'
        opt.new_layer_names = []


        self.model, _ = generate_model(opt)
        self.model.eval()

    
    def extract_features(self, batch):

        # img_data = image.astype(np.float32)
        # img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-5)
        # img_data = np.expand_dims(img_data, axis=(0, 1))
        # img_tensor = torch.tensor(img_data).unsqueeze(0)


        #with torch.no_grad():
        output = self.model(batch)

        return output


    def compute_loss(self, pred, y):
        pred = self.preprocess(pred)
        y = self.preprocess(y)

        pred_feats = self.extract_features(pred)
        y_feats = self.extract_features(y)

        loss_pt = torch.mean(torch.abs(pred_feats - y_feats))

        return loss_pt.item()
 

    def preprocess(self, batch):
        batch = batch.numpy()
        print(f"Feature Extractor here: {batch.shape}")

        arr = batch[:, :, :, 0] # [B, H, W]
        arr = (arr - arr.mean(axis=(1,2), keepdims=True)) / (arr.std(axis=(1,2), keepdims=True) + 1e-5)
        tensor = torch.from_numpy(arr).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]
        
        return tensor


    def visualize_features(self, features):

        num_features = features.shape[1]
        grid_size = math.ceil(math.sqrt(num_features))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))

        for idx in range(num_features):
            row = idx // grid_size
            col = idx % grid_size
            axes[row, col].imshow(features[0, idx, 0, :, :].cpu(), cmap='gray')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig("features_grid.png")
        plt.close()
    



if __name__=="__main__":
    
    image = cv2.imread("img_0.png", cv2.IMREAD_GRAYSCALE)
    print("Input image shape:", image.shape)

    fe = FeatureExtractor()
    features = fe.extract_features(image)
    fe.visualize_features(features)
    