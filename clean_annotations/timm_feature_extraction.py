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

def extract_features_from_bbox(
    img,
    model_type,
    bbox,
    img_transform,
    dino,
    device,
    size_img_new,
    i,
    number_of_first_token_removed,
    patch_size,
    dino_dim=384,
    vis=True,
    input_cell=False
):
    # Get the coordinnates of the bouding boxe 
    bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = img.size
    left = (width - size_img_new) // 2
    top = (height - size_img_new) // 2
    right = left + size_img_new
    bottom = top + size_img_new
    img_cropped = img.crop((left, top, right, bottom))   # Maybe to remove because img_transform might contain cropping too 
    if i == 0 : 
        print("size de img_cropped avant transform",img_cropped.size)
    # --- DINO inference --- 
    with torch.no_grad():
        input_img = img_transform(img_cropped).reshape(1, 3, size_img_new, size_img_new).to(device)
        if i == 0 : 
            print("size de input image", input_img.shape)
        
        ### --- Remove restriction on fixed input image size and change pos_embed vector initialization shape ### 
        
        
        #new_num_patches_init = int(size_img_new//patch_size * size_img_new//patch_size)
        #dino.patch_embed.strict_img_size = False
        #embed_len = new_num_patches_init
        #embed_dim = dino_dim
        #new_pos_embed = torch.randn(1, embed_len, embed_dim) * 0.02
        #dino.pos_embed = nn.Parameter(new_pos_embed.to(device))

        ### --- Compute feature map --- 
        feats = dino.forward_features(input_img)
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
        (seg_crop * 255).astype(np.uint8), (features.shape[0], features.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    seg_downsampled_bool = seg_downsampled > 127  # Shape: [48, 48]
    
    if not input_cell : 
        # Compute the average feature within the bounding box
        masked_features = features[seg_downsampled_bool]
        if masked_features.size == 0:
            print(f"Bounding box {bbox} has zero downsampled area, returning zeros...")
            return np.zeros(dino_dim)
        avg_feature = masked_features.mean(axis=0)
    else : 
        avg_feature = features.mean(axis=0)
    
    # Visualization
    
    if vis:
        plt.figure(figsize=(10, 10))
        plt.imshow(img_cropped)
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
        plt.savefig("input_image")
        img_cell = img_cropped.crop((local_bbox[1], local_bbox[0], local_bbox[3], local_bbox[2]))
        img_cell.save("cropped_cell.png")
        plt.close()
        plt.figure(figsize=(10, 10))
        plt.imshow(seg_downsampled_bool, cmap='gray')
        plt.title("Downsampled Segmentation Mask")
        plt.axis("off")
        plt.savefig("segmentation_mask.png")
        plt.close()

    if i ==  0 : 
        print("shape du feature vector:", avg_feature.shape)
    return avg_feature





def load_image_of_bbox(bbox, center_crop=False):
    """
    Load a 3x3 composite of tiles around the given bounding box and optionally
    return a 512x512 crop centered on the bounding box.

    Args:
        bbox (list or tuple): [imgnumber, row, col, height, width]
            - imgnumber: The image number (e.g., 5 for 'img5')
            - row, col: Top-left coordinates of the bounding box in global coordinates
            - height, width: Size of the bounding box
        center_crop (bool): If True, return a 512x512 crop centered on the bounding box.
                            If False, return the full 768x768 (3x3 tiles) composite.

    Returns:
        PIL.Image: The assembled image with bounding box drawn.
    """
    imgnumber, bbox_row, bbox_col, bbox_height, bbox_width = bbox

    TILE_SIZE = 256
    COMPOSITE_SIZE = TILE_SIZE * 3  # 768x768
    CROP_SIZE = 512
    HALF_CROP = CROP_SIZE // 2

    # Directory with tiles
    ROOT = "/home/franchesoni/walden/"
    imgname = f"img{imgnumber}"
    tiles_dir = Path(ROOT) / f"dataset/{imgname}/tiles"

    # Determine the tile grid start
    # We find the tile that contains the top-left corner of the bbox
    tile_row_start = (bbox_row // TILE_SIZE) * TILE_SIZE
    tile_col_start = (bbox_col // TILE_SIZE) * TILE_SIZE

    # Create a composite image (3x3 tiles)
    composite_image = Image.new(
        "RGB", (COMPOSITE_SIZE, COMPOSITE_SIZE), (255, 255, 255)
    )

    # Load surrounding 3x3 tiles
    for i, drow in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
        for j, dcol in enumerate([-TILE_SIZE, 0, TILE_SIZE]):
            tile_row = tile_row_start + drow
            tile_col = tile_col_start + dcol
            tile_name = f"tile_{int(tile_row)}_{int(tile_col)}.jpeg"
            tile_path = tiles_dir / tile_name

            if tile_path.exists():
                try:
                    tile_image = Image.open(tile_path)
                    if tile_image.mode != "RGB":
                        tile_image = tile_image.convert("RGB")
                    composite_image.paste(
                        tile_image, (j * TILE_SIZE, i * TILE_SIZE)
                    )
                except Exception as e:
                    # If tile loading fails, use a placeholder
                    placeholder = Image.new(
                        "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                    )
                    draw_placeholder = ImageDraw.Draw(placeholder)
                    draw_placeholder.line(
                        (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                    )
                    draw_placeholder.line(
                        (0, placeholder.size[1], placeholder.size[0], 0),
                        fill=(150, 150, 150),
                        width=3,
                    )
                    composite_image.paste(
                        placeholder, (j * TILE_SIZE, i * TILE_SIZE)
                    )
            else:
                # Missing tile placeholder
                placeholder = Image.new(
                    "RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200)
                )
                composite_image.paste(placeholder, (j * TILE_SIZE, i * TILE_SIZE))

    # Draw the bounding box on the composite image
    # Calculate the bounding box coordinates relative to the composite image
    # The composite image's center tile corresponds to (tile_row_start, tile_col_start) in global coords
    # Top-left tile in composite is at (tile_row_start - TILE_SIZE, tile_col_start - TILE_SIZE)
    composite_top_row = tile_row_start - TILE_SIZE
    composite_left_col = tile_col_start - TILE_SIZE

    bbox_row_rel = bbox_row - composite_top_row
    bbox_col_rel = bbox_col - composite_left_col
    bbox_bottom_rel = bbox_row_rel + bbox_height
    bbox_right_rel = bbox_col_rel + bbox_width



    if center_crop:
        # We want to produce a 512x512 crop centered on the bbox center
        bbox_center_row = bbox_row_rel + bbox_height / 2
        bbox_center_col = bbox_col_rel + bbox_width / 2
        # Center the BBox in the crop
        # The BBox center should map to the center of the crop (256, 256)
        left = int(bbox_center_col - HALF_CROP)
        upper = int(bbox_center_row - HALF_CROP)
        right = left + CROP_SIZE
        lower = upper + CROP_SIZE
        # adjust the bbox coordinates according to the new cropped image 
        adjusted_bbox_row = bbox_row_rel - upper
        adjusted_bbox_col = bbox_col_rel - left
        adjusted_bbox_bottom = bbox_bottom_rel - upper
        adjusted_bbox_right = bbox_right_rel - left
        # Ensure we don't go outside the composite image boundaries
        if left < 0:
            right -= left
            left = 0
        if upper < 0:
            lower -= upper
            upper = 0
        if right > COMPOSITE_SIZE:
            left -= right - COMPOSITE_SIZE
            right = COMPOSITE_SIZE
        if lower > COMPOSITE_SIZE:
            upper -= lower - COMPOSITE_SIZE
            lower = COMPOSITE_SIZE

        # Crop the image
        cropped_image = composite_image.crop((left, upper, right, lower))

        w, h = cropped_image.size
        if w < CROP_SIZE or h < CROP_SIZE:
            padded = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
            padded.paste(
                cropped_image, ((CROP_SIZE - w) // 2, (CROP_SIZE - h) // 2)
            )
            cropped_image = padded

        return cropped_image, [int(adjusted_bbox_row), int(adjusted_bbox_col), int(adjusted_bbox_bottom), int(adjusted_bbox_right)]
    else:
        return composite_image, [bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel] 


def main(bbox_file, input_size, dstdir="./features_vectors/", model_name="vit_large_patch14_clip_224", device="cuda",input_cell=False):
    """Re-run Dino on bounding boxes extracted from file and save results as .npy."""
    # 1. --- Load de DINO --- 
    dino_models = timm.list_models('*dino*', pretrained=True)
    print("liste de tous les models dino disponibles sur Timm", dino_models)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing Dino...")
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
    # 3. --- Extraction of features vectors --- 
    bboxes = np.load(bbox_file)
    print("Processing bounding boxes...")
    batch_size = 32
    for i, bbox in enumerate(tqdm.tqdm(bboxes, desc="Processing bounding boxes", unit="bbox")):
        # bbox : img_number, x_min, y_min, height, width
        image, bbox_coord = load_image_of_bbox(bbox, center_crop=center_crop)
        if model_type == "VisionTransformer" : 
            model.set_input_size((size_img_new, size_img_new))
        if i == 0 : 
            print("size_img_new", size_img_new)
            output_file = dstdir / f"X_{bbox_file}_{model_name}_size_{size_img_new}_dim_{dim_feature_vector}.npy"
        feature_vector = extract_features_from_bbox(image, model_type, bbox_coord, img_transform, model, device, size_img_new, i, num_cls_tokens, patch_size, dino_dim=dim_feature_vector,input_cell=input_cell)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        #X_features_np = np.array(X_features, dtype=np.float32)
        #np.save(output_file, X_features_np)
        if (i + 1) % batch_size == 0:
            X_features_np = np.array(X_features, dtype=np.float32)
            if os.path.exists(output_file) : 
                X_all_features = np.load(output_file)
                X_all_features = np.concatenate((X_all_features,X_features_np))
                np.save(output_file, X_all_features)
                print(f"Batch {i+1} saved. All Features shape: {X_all_features.shape}")
            else : 
                np.save(output_file, X_features_np)
                print(f"Batch {i+1} saved. All Features shape: {X_features_np.shape}")
            X_features = []
            X_all_features = []
    X_features_np = np.array(X_features, dtype=np.float32)
    X_all_features = np.load(output_file)
    X_all_features = np.concatenate((X_all_features,X_features_np))
    np.save(output_file, X_all_features)
    print(f"Feature extraction completed. Saved {X_all_features.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)
