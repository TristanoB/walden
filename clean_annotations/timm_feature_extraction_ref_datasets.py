import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision 
from torchvision import datasets
from torchvision.datasets import ImageNet
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

def extract_features_from_bbox(
    img,
    model_type,
    img_transform,
    model,
    device,
    size_img_new,
    i,
    patch_size,
    dino_dim=384,
    vis=True,
    input_cell=False
):

    if i == 0 : 
        print("size de img_cropped avant transform",img.size)
    # --- DINO inference --- 
    width, height = img.size
    left = (width - size_img_new) // 2
    top = (height - size_img_new) // 2
    right = left + size_img_new
    bottom = top + size_img_new
    img_cropped = img.crop((left, top, right, bottom)) 
    with torch.no_grad():
        input_img = img_transform(img_cropped).reshape(1, 3, size_img_new, size_img_new).to(device)
        if i == 0 : 
            print("size de input image", input_img.shape)
        ### --- Compute feature map --- 
        feats = model.forward_features(input_img)
        if i == 0 :     
            print("shape des features extraites par DINO", feats.shape)
        if model_type == "VisionTransformer" : 
            num_patches = int(size_img_new//patch_size)
            if i == 0 : 
                print("num patches", num_patches)
            num_tokens_to_remove = feats.shape[1] - (num_patches * num_patches)
            if i == 0 : 
                print("num_tokens_to_remove",num_tokens_to_remove)
            feats = feats[:,num_tokens_to_remove:,:]
            feats = feats.reshape(1, num_patches, num_patches, dino_dim)
        elif model_type == "ConvNeXt" : 
            shape_feats = feats.shape
            feats = feats.reshape(1, dino_dim, shape_feats[2]*shape_feats[2])
            feats = feats.permute((0,2,1))
        if i == 0 : 
            print("shape des features reshape", feats.shape)
    features = feats.squeeze(0).cpu().numpy()  # Shape: [num_patches, num_patches, dino_dim]
    avg_feature = features.reshape((features.shape[0]*features.shape[1],-1)).mean(axis=0)
    if vis:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_cropped)
        plt.axis("off")
        plt.savefig("input_image")
    if i ==  0 : 
        print("shape du feature vector:", avg_feature.shape)
    return avg_feature

def main(dataset_name, input_size, dstdir="./features_vectors/", model_name="vit_large_patch14_clip_224", device="cuda",input_cell=False):
    """extract features vectors from images of reference dataset"""
    # 1. --- Load du model --- 
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing model...")
    # Load the model using timm
    model = timm.create_model(model_name, pretrained=True)
    model_type = model.__class__.__name__
    print(f"Type d'architecture : {model_type}")
    size_img_new = input_size
    if model_type == "VisionTransformer" : 
        patch_size = model.patch_embed.patch_size[0]  # 16
        print("patch_size", patch_size)
        num_cls_tokens = model.cls_token.shape[1] if model.cls_token is not None else 0
        print("num_cls_tokens", num_cls_tokens)
    else : 
        patch_size = None
        num_cls_tokens = None
    center_crop = True
    model = model.to(device)
    model = model.eval()
    dim_feature_vector = model.num_features
    print("dim_feature_vector",dim_feature_vector)
    # 2. --- Image initialization ---
    data_config = timm.data.resolve_model_data_config(model)
    img_transform = timm.data.create_transform(**data_config, is_training=False)
    mean = data_config['mean']
    std = data_config['std']
    print("Image transform implémentée de base avec le modèle : ", img_transform)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
      transforms.Normalize(mean=mean, std=std),
    ])
    print(f"transformation appliquée à l'image : {img_transform}")
    dstdir = Path(dstdir)
    X_features = []
    y_labels = []

    #imagenet_dataset = ImageNet(root="path_to_imagenet", split="train", transform=img_transform)
    dataset = datasets.CIFAR10(root="datasets/cifar10", train=True, download=True)
    batch_size = 32

    # 3. --- Extraction of features vectors --- 
    for i, (image, label) in enumerate(tqdm.tqdm(dataset, desc="Processing images", unit="images")):
        # bbox : img_number, x_min, y_min, height, width
        if model_type == "VisionTransformer" : 
            model.set_input_size((size_img_new, size_img_new))
        if i == 0 : 
            print("size_img_new", size_img_new)
            output_file = dstdir / f"X_{dataset_name}_{model_name}_size_{size_img_new}_dim_{dim_feature_vector}.npy"
        feature_vector = extract_features_from_bbox(image, model_type, img_transform, model, device, size_img_new, i, patch_size, dino_dim=dim_feature_vector,input_cell=input_cell)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        y_labels.append(label)

        if (i + 1) % batch_size == 0:
            X_features_np = np.array(X_features, dtype=np.float32)
            np.save(output_file, X_features_np)
            print(f"Batch {i+1} saved. Features shape: {X_features_np.shape}")
            X_features = []
        np.save(output_file, X_features_np)
        y_labels_np = np.array(y_labels, dtype=np.int64)
        np.save(dstdir / f"y_{dataset_name}_{model_name}_size_{size_img_new}_dim_{dim_feature_vector}.npy", y_labels_np)

    print(f"Feature extraction completed. Saved {X_features_np.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)