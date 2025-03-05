import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
from pathlib import Path
from PIL import Image
import fire
import tqdm
import timm
import cv2
import os
import matplotlib.pyplot as plt
import h5py
import csv
import shutil

def extract_features_from_bbox(
    img,
    bbox,
    img_transform,
    dino,
    device,
    size_img_new,
    i,
    dino_dim=384,
    vis=True,
):
    bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = img.size
    left = (width - size_img_new) // 2
    top = (height - size_img_new) // 2
    right = left + size_img_new
    bottom = top + size_img_new
    img_downsampled = img.crop((left, top, right, bottom))   # Maybe to remove because img_transform might contain cropping too 
    if i == 0 : 
        print("size de img_cropped avant transform",img_downsampled.size)
    # --- DINO inference --- 
    with torch.no_grad():
        print("dino model used", dino)
        print("img size dans dino", dino.patch_embed.img_size)
        print("dino.patch_embed.num_patches", dino.patch_embed.num_patches)
        input_img = img_transform(img_downsampled)
        print("size de input img après transform : ", input_img.size())
        input_img = input_img.reshape(1, 3, size_img_new, size_img_new).to(device)
        if i == 0 : 
            print("size de input image", input_img.shape)
        feats = dino.forward_features(input_img)
        print("keys dans feats", feats.keys())
        feats = feats["x_norm_patchtokens"]
        if i == 0 :     
            print("shape des features extraites par DINO", feats.shape)
        patch_dim = int(np.sqrt(feats.shape[1]))
        if i == 0 : 
            print("patch_dim", patch_dim)
        feats = feats.reshape(
            1, patch_dim, patch_dim, dino_dim
        )
        if i == 0 : 
            print("shape des features reshape", feats.shape)
    features = feats.squeeze(0).cpu().numpy()  # Shape: [48, 48, dino_dim]
    adjusted_bbox_row_rel = bbox_row_rel - top
    adjusted_bbox_col_rel = bbox_col_rel - left
    adjusted_bbox_bottom_rel = bbox_bottom_rel - top
    adjusted_bbox_right_rel = bbox_right_rel - left
    local_bbox = (
        adjusted_bbox_row_rel, adjusted_bbox_col_rel, adjusted_bbox_bottom_rel, adjusted_bbox_right_rel
    )
    # Create segmentation mask for the bounding box
    seg_crop = np.zeros((size_img_new, size_img_new), dtype=np.uint8)
    seg_crop[local_bbox[0]:local_bbox[2], local_bbox[1]:local_bbox[3]] = 1
    
    # Downsample the segmentation to match the DINO feature map size
    seg_downsampled = cv2.resize(
        (seg_crop * 255).astype(np.uint8), (patch_dim, patch_dim), interpolation=cv2.INTER_LINEAR
    )
    seg_downsampled_bool = seg_downsampled > 127  # Shape: [48, 48]
    
    # Compute the average feature within the bounding box
    masked_features = features[seg_downsampled_bool]
    if masked_features.size == 0:
        print(f"Bounding box {bbox} has zero downsampled area, returning zeros...")
        return np.zeros(dino_dim)
    
    avg_feature = masked_features.mean(axis=0)
    
    # Visualization
    
    if vis:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_downsampled)
        plt.gca().add_patch(
            plt.Rectangle(
                (local_bbox[1], local_bbox[0]),  # Position (x, y)
                local_bbox[3] - local_bbox[1],  # Largeur = bbox_right - bbox_col (+10 de chaque côté)
                local_bbox[2] - local_bbox[0],  # Hauteur = bbox_bottom - bbox_row (+10 de chaque côté)
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
        )
        plt.axis("off")
        plt.savefig("out")
        img_cell = img_downsampled.crop((local_bbox[1], local_bbox[0], local_bbox[3], local_bbox[2]))
        img_cell.save("out_cells.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(seg_downsampled_bool, cmap='gray')
        plt.title("Downsampled Segmentation Mask")
        plt.axis("off")
        plt.savefig("segmentation_mask.png")
    return avg_feature


def load_random_image_with_bbox(image_size=(512, 512), bbox_size=(100, 100)):
    image = np.random.randint(0, 256, (image_size[0], image_size[1], 3), dtype=np.uint8)
    image_pil = Image.fromarray(image)
    x_min = np.random.randint(0, image_size[1] - bbox_size[1])
    y_min = np.random.randint(0, image_size[0] - bbox_size[0])
    x_max = x_min + bbox_size[1]
    y_max = y_min + bbox_size[0]
    bbox = (x_min, y_min, x_max, y_max)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return image_pil, bbox





import timm
import torch

def main(dstdir, dino_model="dinov2_vits14_reg", dino_size="small", device="cuda"):
    """Re-run Dino on random bounding boxes extracted from random images to test the input sizes"""
    # 1. --- Load de DINO --- 
    print("Initializing Dino...")
    if dino_model == "dinov2_vits14_reg":
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg") # Small
        dino = dino.to(device)
        dino_dim = 384
    elif dino_model == "dinov2_vitb14_reg" : 
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")  # Base
        dino = dino.to(device)
        dino_dim = 768  
    elif dino_model == "dinov2_vitl14_reg" : 
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")  # Large
        dino = dino.to(device)
        dino_dim = 1024  
    elif dino_model == "dinov2_vitg14_reg" : 
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14_reg")  # Giant
        dino = dino.to(device)
        dino_dim = 1536   
    center_crop = True
    size_img_new = 224 
    print(f"Loaded {dino_model} with dino_dim={dino_dim}, center_crop={center_crop}, size_img_new={size_img_new}")
    dino.eval()
    # 2. --- Image initialization ---
    data_config = timm.data.resolve_model_data_config(dino)
    img_transform = timm.data.create_transform(**data_config, is_training=False)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"transformation appliquée à l'image : {img_transform}")
    dstdir = Path(dstdir)
    X_features = []
    # 3. --- Extraction of features vectors --- 
    print("Processing bounding boxes...")
    for i in range(1):
        # bbox : img_number, x_min, y_min, height, width
        image, bbox_coord = load_random_image_with_bbox()
        print("dino.patch_embed.img_size", dino.patch_embed.img_size)
        size_img_new = 308
        if i == 0 : 
            print("size_img_new", size_img_new)
            output_file = dstdir / f"X_labeled_test.npy"
        feature_vector = extract_features_from_bbox(image, bbox_coord, img_transform, dino, device, size_img_new, i, dino_dim=dino_dim)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        X_features_np = np.array(X_features, dtype=np.float32)
        np.save(output_file, X_features_np)
    print(f"Feature extraction completed. Saved {X_features_np.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)
