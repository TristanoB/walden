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
    img768,
    bbox,
    img_transform,
    dino,
    device,
    dino_dim=384,
    vis=True,
):
    """
    Extract feature vector from the bounding box region in the image using DINO.
    
    Args:
        img768 (PIL.Image): Image containing the bounding box at its center.
        bbox (tuple): Bounding box coordinates (imgnumber, bbox_row, bbox_col, bbox_height, bbox_width).
        img_transform (callable): Image transformation pipeline.
        device (torch.device): Device for computation.
        dino (torch.nn.Module): Pre-trained DINO model.
        dino_dim (int): Feature dimension of DINO.
        vis (bool): Whether to visualize the bounding box.

    Returns:
        np.ndarray: Extracted feature vector.
    """
    bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel = bbox[0], bbox[1], bbox[2], bbox[3]
    size_img = 224
    new_size = (size_img, size_img)  # 756 est le multiple de 14 le plus proche de 768
    img_cropped = img768.resize(new_size, Image.BILINEAR)   
    # --- DINO inference --- 
    with torch.no_grad():
        input_img = img_transform(img_cropped).reshape(1, 3, size_img, size_img).to(device)
        feats = dino.forward_features(input_img)["x_norm_patchtokens"]
        feats = feats.reshape(
            1, 54, 54, dino_dim
        )
    features = feats.squeeze(0).cpu().numpy()  # Shape: [48, 48, dino_dim]
    local_bbox = (
        bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel
    )
    # Create segmentation mask for the bounding box
    seg_crop = np.zeros((size_img, size_img), dtype=np.uint8)
    seg_crop[local_bbox[0]:local_bbox[2], local_bbox[1]:local_bbox[3]] = 1
    # Downsample the segmentation to match the DINO feature map size
    seg_downsampled = cv2.resize(
        (seg_crop * 255).astype(np.uint8), (54, 54), interpolation=cv2.INTER_LINEAR
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
        # Visualize the bounding box on the full image
        plt.figure(figsize=(10, 10))
        plt.imshow(img768)
        plt.gca().add_patch(
            plt.Rectangle(
                (bbox_col_rel - 10, bbox_row_rel - 10),  # Position (x, y)
                bbox_right_rel - bbox_col_rel + 20,  # Largeur = bbox_right - bbox_col (+10 de chaque côté)
                bbox_bottom_rel - bbox_row_rel + 20,  # Hauteur = bbox_bottom - bbox_row (+10 de chaque côté)
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
        )
        plt.axis("off")
        plt.savefig("out")

        # Visualize the cropped cell (just the patch for debugging)
        img_cell = img768.crop((bbox_col_rel - 10, bbox_row_rel - 10, bbox_right_rel + 10, bbox_bottom_rel + 10))
        img_cell.save("out_cells.png")

        # Visualize the segmentation mask applied to the DINO feature map
        plt.figure(figsize=(10, 10))
        plt.imshow(seg_downsampled_bool, cmap='gray')
        plt.title("Downsampled Segmentation Mask")
        plt.axis("off")
        plt.savefig("segmentation_mask.png")


    
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

        return cropped_image, [bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel] 
    else:
        return composite_image, [bbox_row_rel, bbox_col_rel, bbox_bottom_rel, bbox_right_rel] 


def main(dstdir, bbox_file, dino_size="small", device="cuda"):
    """Re-run Dino on bounding boxes extracted from .pkl file and save results as .npy."""
    # 1. --- Load de DINO --- 
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("Initializing Dino...")
    # Load DINO using timm
    if dino_size == "small":
        dino = timm.create_model("vit_small_patch16_224", pretrained=True)  # Use the small variant (adjust to the appropriate DINO variant)
        dino = dino.to(device)
        dino_dim = 384
    elif dino_size == "big": 
        # For big model, use a larger variant, e.g., vit_large_patch32_224
        dino = timm.create_model("vit_large_patch32_384", pretrained=True)
        dino = dino.to(device)
        dino_dim = 768  # Change dino_dim accordingly for the big mod
    else:
        raise ValueError(f"{dino_size} not recognized, should be ['small']")
    dino.eval()
    # 2. --- Image initialization ---
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dstdir = Path(dstdir)
    output_file = dstdir / "X_labeled_new.npy"
    X_features = []
    # 3. --- Extraction of features vectors --- 
    bboxes = np.load(bbox_file)
    print("Processing bounding boxes...")
    for bbox in tqdm.tqdm(bboxes, desc="Processing bounding boxes", unit="bbox"):
        # bbox : img_number, x_min, y_min, height, width
        image, bbox_coord = load_image_of_bbox(bbox, center_crop=False)
        feature_vector = extract_features_from_bbox(image, bbox_coord, img_transform, dino, device, dino_dim=dino_dim)
        # 4. --- Save the feature vectors array to the .npy file ---
        X_features.append(feature_vector)
        X_features_np = np.array(X_features, dtype=np.float32)
        np.save(output_file, X_features_np)
    print(f"Feature extraction completed. Saved {X_features.shape[0]} feature vectors.")

if __name__ == "__main__":
    fire.Fire(main)
