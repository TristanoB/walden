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
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import fire
import io
from matplotlib.backends.backend_pdf import PdfPages


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



def figure_to_high_res_image(fig, dpi=100):
    """Convertit une figure matplotlib en image haute résolution."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return Image.open(buf)

def main(dstdir, bbox_file):
    """Re-run Dino on bounding boxes extracted from file and save results as .npy."""
    
    # 1. --- Initialisation ---
    dstdir = Path(dstdir)
    bboxes = np.load(bbox_file)
    
    with open('true_formatted_cell_dataset.pkl', 'rb') as f:
        labeled_data = pickle.load(f)
    class_names = labeled_data['class_names']
    y_labeled = np.load('y_labeled.npy')

    # 2. --- Dictionnaire des images par classe ---
    images_by_class = {class_name: [] for class_name in class_names}

    # 3. --- Extraction et ajout des bounding boxes ---
    print("Processing bounding boxes...")
    for i, bbox in enumerate(tqdm.tqdm(bboxes, desc="Processing bounding boxes", unit="bbox")):
        image, bbox_coord = load_image_of_bbox(bbox, center_crop=True)
        label = y_labeled[i]
        class_name_cell = class_names[label]

        # 4. --- Création de la figure avec la bbox ---
        fig, ax = plt.subplots(figsize=(3, 3), dpi=300)  # Figure haute résolution
        ax.imshow(image)
        ax.add_patch(
            plt.Rectangle(
                (bbox_coord[1], bbox_coord[0]),  # Position (x, y)
                bbox_coord[3] - bbox_coord[1],  # Largeur
                bbox_coord[2] - bbox_coord[0],  # Hauteur
                edgecolor="red",
                facecolor="none",
                lw=2,
            )
        )
        ax.set_title(f"{class_name_cell}_index_{i}", fontsize=8)
        ax.axis("off")

        # Convertir la figure en image haute résolution et stocker
        high_res_img = figure_to_high_res_image(fig)
        images_by_class[class_name_cell].append(high_res_img)
        plt.close(fig)  # Ferme la figure pour éviter la surcharge mémoire

    # 5. --- Création du PDF ---
    output_file = dstdir / "all_cells_images.pdf"
    with PdfPages(output_file) as pdf:
        total_images = sum(len(images) for images in images_by_class.values())
        
        with tqdm.tqdm(total=total_images, desc="Saving images to PDF", unit="image") as pbar:
            for class_name, images in images_by_class.items():
                num_images = len(images)
                num_per_page = 20
                num_pages = (num_images + num_per_page - 1) // num_per_page

                for page_idx in range(num_pages):
                    fig, axes = plt.subplots(5, 4, figsize=(15, 18), dpi=300)  # Grille 5x4 avec haute résolution
                    axes = axes.flatten()

                    start_idx = page_idx * num_per_page
                    end_idx = min(start_idx + num_per_page, num_images)

                    for ax, img in zip(axes, images[start_idx:end_idx]):
                        ax.imshow(img, aspect='auto', interpolation='nearest')  # Affichage sans perte de qualité
                        ax.set_title(class_name, fontsize=14)
                        ax.axis("off")

                    for ax in axes[end_idx - start_idx:]:
                        ax.axis("off")  # Cache les axes restants s'il y en a moins que 20

                    pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    pbar.update(end_idx - start_idx)

    print(f"✅ PDF généré : {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
