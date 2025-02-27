import torch
import numpy as np
import torchvision.transforms as transforms
import pickle
from pathlib import Path
from PIL import Image
import fire

print("importing")
import tqdm
from pathlib import Path
import torch
import numpy as np
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import matplotlib.pyplot as plt
import h5py
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.utils.amg import build_all_layer_point_grids
import csv
import shutil

def extract_features(
    srcdir,
    dstdir,
    bbox, 
    tile_row,
    tile_col,
    center_bbox,
    img_transform,
    device,
    dino,
    dino_dim,
    vis,
):
    img1024, global_row, global_col = load_1024(srcdir, tile_row, tile_col)
    
    
   

    # --- Crop central region for DINO ---
    img644 = img1024.crop((190, 190, 834, 834))
    with torch.no_grad():
        input_img = img_transform(img644).reshape(1, 3, 644, 644).to(device)
        feats = dino.forward_features(input_img)["x_norm_patchtokens"].reshape(
            1, 46, 46, dino_dim
        )
    
    # --- Extract features for each mask ---
    features = feats.squeeze(0).cpu().numpy()  # Shape: [46, 46, dino_dim]
    for mask in filtered_masks:
        segmentation = mask["segmentation"]  # Shape: [1024, 1024]
        
        # Crop the segmentation to the center crop coordinates
        seg_crop = segmentation[190:834, 190:834]  # Shape: [644, 644]
        
        # Downsample the segmentation to match the DINO feature map size
        seg_downsampled = cv2.resize(
            (seg_crop * 255).astype(np.uint8), (46, 46), interpolation=cv2.INTER_LINEAR
        )
        seg_downsampled_bool = seg_downsampled > 127  # Shape: [46, 46]

        # Compute the average feature within the mask
        masked_features = features[seg_downsampled_bool]
        if masked_features.size == 0:
            print(
                f"mask at {mask['global_bbox']} has zero downsampled area, skipping..."
            )
            continue
        avg_feature = masked_features.mean(axis=0)

        append_to_h5(avg_feature, dino_dim, dstdir / "features.h5")
        append_to_csv(mask["global_bbox"], dstdir / "global_bboxes.txt")
    
    if vis:
        plt.figure(figsize=(20, 20))
        plt.imshow(img1024)
        show_anns(filtered_masks)
        plt.axis("off")
        plt.savefig("out.png")
        plt.close()




def extract_features_from_bbox(image, bbox, img_transform, dino, device):
    """Extract features from an image within the given bounding box."""
    label, x_min, y_min, height, width = bbox  
    cropped_img = image.crop((y_min, x_min, y_min + width, x_min + height))
    
    with torch.no_grad():
        input_img = img_transform(cropped_img).unsqueeze(0).to(device)
        feats = dino.forward_features(input_img)["x_norm_patchtokens"].squeeze(0).cpu().numpy()
    
    return feats.mean(axis=0), label  # Retourne le vecteur de features et son label

def main(srcdir, dstdir, bbox_file, dino_size="small", device="cuda"):
    """Re-run Dino on bounding boxes extracted from .pkl file and save results as .npy."""
    
    # --- Load de DINO --- 
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing Dino...")
    if dino_size == "small":
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg", force_reload=False)
        dino = dino.to(device)
        dino_dim = 384
    else:
        raise ValueError(f"{dino_size} not recognized, should be ['small']")
    dino.eval()
    
    # --- Image initialization ---
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    srcdir, dstdir = Path(srcdir), Path(dstdir)
    bboxes = np.load(bbox_file)
    X_features = []
    y_labels = []

    # --- Extraction of features vectors --- 
    print("Processing bounding boxes...")
    for bbox in bboxes:
        # bbox : label, x_min, y_min, height, width
        img_path = srcdir / f"tiles/tile_{bbox[0]}_{bbox[1]}.jpeg"
        if not img_path.exists():
            print(f"Image {img_path} not found, skipping...")
            continue
        
        image = Image.open(img_path)
        feature_vector, label = extract_features_from_bbox(image, bbox, img_transform, dino, device)
        
        X_features.append(feature_vector)
        y_labels.append(label)

    # --- Save des arrays features vectors/labels --- 
    X_features = np.array(X_features, dtype=np.float32)
    y_labels = np.array(y_labels, dtype=np.int32)
    np.save(dstdir / "X_labeled.npy", X_features)
    np.save(dstdir / "y_labels.npy", y_labels)

    print(f"Feature extraction completed. Saved {X_features.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)
