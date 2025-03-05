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
import torch.nn.functional as F
from urllib.request import urlopen
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch >= 2.31.0, timm >= 1.0.15


def extract_features_from_bbox(
    image,
    bbox,
    preprocess,
    model,
    device,
    size_img_new,
    i,
    number_of_first_token_removed,
    patch_size,
    dim_features,
    vis=True,
):
    # Get the coordinnates of the bouding boxe 
    bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = image.size
    left = (width - size_img_new) // 2
    top = (height - size_img_new) // 2
    right = left + size_img_new
    bottom = top + size_img_new
    img_cropped = image.crop((left, top, right, bottom))  
    adjusted_bbox_row_rel = bbox_row_rel - top
    adjusted_bbox_col_rel = bbox_col_rel - left
    adjusted_bbox_bottom_rel = bbox_bottom_rel - top
    adjusted_bbox_right_rel = bbox_right_rel - left
    local_bbox = (
        adjusted_bbox_row_rel, adjusted_bbox_col_rel, adjusted_bbox_bottom_rel, adjusted_bbox_right_rel
    )
    cell_image = img_cropped.crop((local_bbox[0], local_bbox[1], local_bbox[3], local_bbox[2]))
    cell_image = cell_image.resize((size_img_new, size_img_new))
    if i == 0 : 
        print("size de img_cropped avant transform",img_cropped.size)
    # --- DINO inference --- 
    with torch.no_grad():
        input_img = preprocess(cell_image).unsqueeze(0).to(device)
        if i == 0 : 
            print("size de input image", input_img.shape)
        ### --- Compute feature map --- 
        feats = model.encode_image(input_img, normalize=True)
        if i == 0 :     
            print("shape des features extraites par le modèle", feats.shape)
    features = feats.squeeze(0).cpu().numpy()  # Shape:  [dino_dim]
        
    
    if vis:
        plt.figure(figsize=(10, 10))
        plt.imshow(cell_image)
        plt.axis("off")
        plt.savefig("out")

    return features





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


def main(dstdir, bbox_file, model_name="ViT-L-16-SigLIP2-512", device="cuda"):
    """Re-run Dino on bounding boxes extracted from file and save results as .npy."""
    # 1. --- Load de DINO --- 
    dino_models = timm.list_models('*mae*', pretrained=True)
    print("liste de tous les models dino disponibles sur Timm", dino_models)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing Dino...")
    # Load DINO using timm
    
    if model_name == "ViT-L-16-SigLIP2-512":
        model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP2-512')
        tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP2-512')
        model = model.to(device)
        model = model.eval()
        dim_features = 384
        center_crop = True
        number_of_first_token_removed = 1
        patch_size = 16



    else:
        raise ValueError(f"{model_name} not recognized, should be ['small']")
    model.eval()
    # 2. --- Image initialization ---#    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    dstdir = Path(dstdir)
    X_features = []
    # 3. --- Extraction of features vectors --- 
    bboxes = np.load(bbox_file)
    print("Processing bounding boxes...")
    for i, bbox in enumerate(tqdm.tqdm(bboxes, desc="Processing bounding boxes", unit="bbox")):
        # bbox : img_number, x_min, y_min, height, width
        image, bbox_coord = load_image_of_bbox(bbox, center_crop=center_crop)
        size_img_new = 512
        #model.set_input_size((size_img_new, size_img_new))
        if i == 0 : 
            print("size_img_new", size_img_new)
            output_file = dstdir / f"X_labeled_{model_name}_input_size_{size_img_new}.npy"
        feature_vector = extract_features_from_bbox(image, bbox_coord, preprocess, model, device, size_img_new, i, number_of_first_token_removed, patch_size, dim_features)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        X_features_np = np.array(X_features, dtype=np.float32)
        np.save(output_file, X_features_np)
    print(f"Feature extraction completed. Saved {X_features_np.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)