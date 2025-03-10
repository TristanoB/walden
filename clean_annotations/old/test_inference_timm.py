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
import torch.nn as nn


def resize_pos_embed(model, x):
    pos_embed = model.pos_embed[:, 1:]  # Supprime le token CLS
    num_patches_new = x.shape[1]  # Nombre de patches après PatchEmbed

    if num_patches_new != pos_embed.shape[1]:
        from timm.models.vision_transformer import resize_pos_embed
        model.pos_embed = torch.nn.Parameter(resize_pos_embed(model.pos_embed, num_patches_new))

def extract_features_from_bbox(
    img,
    bbox,
    img_transform,
    dino,
    device,
    size_img_new,
    i,
    number_of_first_token_removed,
    dino_dim=384,
    vis=True,
):
    """
    Extract feature vector from the bounding box region in the image using DINO.
    
    Args:
        img (PIL.Image): Image containing the bounding box at its center.
        bbox (tuple): Bounding box coordinates (imgnumber, bbox_row, bbox_col, bbox_height, bbox_width).
        img_transform (callable): Image transformation pipeline.
        device (torch.device): Device for computation.
        dino (torch.nn.Module): Pre-trained DINO model.
        dino_dim (int): Feature dimension of DINO.
        vis (bool): Whether to visualize the bounding box.

    Returns:
        np.ndarray: Extracted feature vector.
    """
    # Get the coordinnates of the bouding boxe 
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
        print("DINO dynamic image size", dino.dynamic_img_size)
        #dino.dynamic_img_size = True
        #dino.strict_img_size = False
        #print("modele dino", dino)
        print("img size dans dino", dino.patch_embed.img_size)
        dino.patch_embed.strict_img_size = False
        print("strict image_size dans patch embedded de dino", dino.patch_embed.strict_img_size)
        print("dino.patch_embed.num_patches", dino.patch_embed.num_patches)
        embed_len = dino.patch_embed.num_patches
        embed_dim = dino_dim
        new_pos_embed = torch.randn(1, embed_len, embed_dim) * 0.02
        dino.pos_embed = nn.Parameter(new_pos_embed.to(device))
        input_img = img_transform(img_downsampled).reshape(1, 3, size_img_new, size_img_new).to(device)
        if i == 0 : 
            print("size de input image", input_img.shape)
        feats = dino.forward_features(input_img)
        if i == 0 :     
            print("shape des features extraites par DINO", feats.shape)
        feats = feats[:,number_of_first_token_removed:,:]
        num_patches = int(np.sqrt(feats.shape[1]))
        if i == 0 : 
            print("num patches", num_patches)
        feats = feats.reshape(
            1, num_patches, num_patches, dino_dim
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

def main(dstdir, dino_model="vit_small_patch14_reg4_dinov2.lvd142m", dino_size="small", device="cuda"):
    """Re-run Dino on random bounding boxes extracted from random images to test the input sizes"""
    # 1. --- Load de DINO --- 
    dino_models = timm.list_models('*dino*', pretrained=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing DINO...")
    dino = timm.create_model(dino_model,pretrained=True).to(device).eval()
    dino_dim_map = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536
    }
    dino_dim = dino_dim_map[dino_size] if dino_size else None
    center_crop = True
    size_img_new = 224 
    number_of_first_token_removed = 5 if "reg4" in dino_model else 1 
    print(f"Loaded {dino_model} with dino_dim={dino_dim}, center_crop={center_crop}, size_img_new={size_img_new}, tokens_removed={number_of_first_token_removed}")
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
        print("dino.patch_embed.num_patches", dino.patch_embed.num_patches)
        
        size_img_new = 294
        dino.patch_embed.num_patches = int(size_img_new//14 * size_img_new//14)
        print("new dino.patch_embed.num_patches", dino.patch_embed.num_patches)
        if i == 0 : 
            print("size_img_new", size_img_new)
            output_file = dstdir / f"X_labeled_test.npy"
        feature_vector = extract_features_from_bbox(image, bbox_coord, img_transform, dino, device, size_img_new, i, number_of_first_token_removed, dino_dim=dino_dim)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        X_features_np = np.array(X_features, dtype=np.float32)
        np.save(output_file, X_features_np)
    print(f"Feature extraction completed. Saved {X_features_np.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)
