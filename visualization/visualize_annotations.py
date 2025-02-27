from pathlib import Path
from PIL import Image, ImageDraw

ROOT = "/home/franchesoni/walden/"

def load_image_with_bbox(bbox, center_crop=False):
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
    # Unpack the bounding box
    imgnumber, bbox_row, bbox_col, bbox_height, bbox_width = bbox

    TILE_SIZE = 256
    COMPOSITE_SIZE = TILE_SIZE * 3  # 768x768
    CROP_SIZE = 512
    HALF_CROP = CROP_SIZE // 2

    # Directory with tiles
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
                draw_placeholder = ImageDraw.Draw(placeholder)
                draw_placeholder.line(
                    (0, 0) + placeholder.size, fill=(150, 150, 150), width=3
                )
                draw_placeholder.line(
                    (0, placeholder.size[1], placeholder.size[0], 0),
                    fill=(150, 150, 150),
                    width=3,
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

    draw = ImageDraw.Draw(composite_image)
    draw.rectangle(
        [bbox_col_rel-10, bbox_row_rel-10, bbox_right_rel+10, bbox_bottom_rel+10],
        outline="green",
        width=3,
    )

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

        # If needed, pad to ensure exactly 512x512
        w, h = cropped_image.size
        if w < CROP_SIZE or h < CROP_SIZE:
            padded = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (255, 255, 255))
            padded.paste(
                cropped_image, ((CROP_SIZE - w) // 2, (CROP_SIZE - h) // 2)
            )
            cropped_image = padded

        return cropped_image
    else:
        return composite_image


def visualize_annotations(annotations, bboxes, positive_only=False):
    # Helper: find next index matching condition.
    def next_valid_index(start):
        i = start
        while i < len(annotations):
            if not positive_only or annotations[i][1]:
                return i
            i += 1
        return None

    current_ind = next_valid_index(0)
    if current_ind is None:
        print("No matching samples.")
        return

    # Load current image.
    img_ind, is_positive = annotations[current_ind]
    bbox = bboxes[img_ind]
    current_img = load_image_with_bbox(bbox, center_crop=True)

    next_ind = next_valid_index(current_ind + 1)
    if next_ind is not None:
        next_img_ind, _ = annotations[next_ind]
        next_bbox = bboxes[next_img_ind]
        next_img = load_image_with_bbox(next_bbox, center_crop=True)
    else:
        next_img = None

    while True:
        current_img.save("tmp.png")
        print(f"ind: {current_ind} pos?: {annotations[current_ind][1]}")
        input("next? ")

        if next_ind is None:
            print("End of samples.")
            break

        # Prefetch: make next image the current image.
        current_ind = next_ind
        current_img = next_img

        next_ind = next_valid_index(current_ind + 1)
        if next_ind is not None:
            next_img_ind, _ = annotations[next_ind]
            next_bbox = bboxes[next_img_ind]
            next_img = load_image_with_bbox(next_bbox, center_crop=True)
        else:
            next_img = None

import pickle
import csv

print('loading cell dataset...')
with open("cell_dataset.pkl", "rb") as f:
    bboxes = pickle.load(f)["bboxes"]
    print('shape de bboxes', bboxes.shape)

print('loading annotation file...')
annotation_file = "annotations_lymphoplasmocyte.csv"
with open(annotation_file, 'r') as f:
    reader = csv.reader(f)
    # skip header
    annotations = [(int(row[0]), bool(int(row[1]))) for row in reader]
# reverse annotations
annotations = annotations[::-1]
visualize_annotations(annotations, bboxes, positive_only=False)

# this is a tool to visualize positive or all annotations of each class