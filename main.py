import cv2
import os
import numpy as np
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import torch.nn.functional as F

clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

from src.util import find_color_cube, load_cv2_img, mask_bordered_region, create_color_mask, mask_straight_lines, show_img

DEBUG = False


border_color_rgb = [88, 93, 96]
security_item_rgb = [4, 21, 42]

border_color_2_rgb = [73, 81, 84]

def get_image_text_similarity(images, text):
    # Preprocess the text
    text_inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    
    # Encode the text
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    
    image_scores = []
    
    for image in images:
        # Preprocess the image
        image_inputs = clip_processor(images=image, return_tensors="pt")
        
        # Encode the image
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
        
        # Calculate similarity
        similarity = torch.matmul(image_features, text_features.T).squeeze().item()
        image_scores.append(similarity)
    
    return image_scores

def security_check_img_boundary(image):
    bordered_img = mask_bordered_region(image, border_color_rgb)
    x, y, w, h = find_color_cube(bordered_img,border_color_rgb, 10)
    return x,y,w,h

def does_cell_have_artifact(grid_mask, cell_cord): #xy
    h, w = 10, 20
    x,y = cell_cord

    artifact = grid_mask[y:y+h,x:x+w]
    return not np.all(artifact == 0)

def is_occupied(occupied_cells, x, y):
    for start, end in occupied_cells:
        start_x, start_y = start
        end_x, end_y = end
        
        if start_x <= x + 1 <= end_x and start_y <= y + 1 <= end_y:
            return True
    return False

def group_object_cells(start_coord, occupied, line_mask, grid_mask, size, width, height, it_x=1,it_y=1):
    x, y = start_coord

    # Calculate the right and bottom coordinates
    right_x = min(x + size, width)
    bottom_y = min(y + size, height)

    # Initialize final coordinates
    final_x, final_y = x, y

    w, h, = it_x, it_y
    # Check the cell to the right
    if right_x+10 <= width and not does_cell_have_artifact(grid_mask, (right_x, y)) and not is_occupied(occupied, right_x, y):
        line_region = line_mask[y+15:y+25,right_x:right_x+5]
        if np.all(line_region == 0):
            final_x, _, bw, bh = group_object_cells((right_x, y), occupied, line_mask, grid_mask, size, width, height, it_x=w+1, it_y=h)
            w = bw
            h = bh

    # Check the cell below
    if bottom_y+10 <= height and not does_cell_have_artifact(grid_mask, (x, bottom_y)) and not is_occupied(occupied, x, bottom_y):
        line_region = line_mask[bottom_y:bottom_y+5,x+15:x+25]
        
        if np.all(line_region == 0):
            _, final_y, bw, bh = group_object_cells((x, bottom_y), occupied, line_mask, grid_mask, size, width, height, it_x=w, it_y=h+1)
            w = bw
            h = bh

    return final_x, final_y, w, h

def draw_grid(image, grid_mask,line_mask, size=84):
    height, width = image.shape[:2]
    
    grid_image = image.copy()
    
    row_count = height // size
    col_count = width // size

    occupied_cells = []

    print((row_count+1),(col_count+1))

    # iterate every cell, put 2 lines
    for cell in range((row_count+1)*(col_count+1)):
        row = cell // (col_count + 1)
        col = cell % (col_count + 1)
        
        x = col * size
        y = row * size
        if not does_cell_have_artifact(grid_mask, (x,y)):
            continue

        x2, y2, w, h = group_object_cells((x,y), occupied_cells,line_mask, grid_mask, size, width, height)

        occupied_cells.append(((x,y), (x2+size, y2+size)))
        cv2.rectangle(grid_image, (x,y), (x2+size, y2+size), (0,255,0), 2) 
        
    return grid_image, occupied_cells

def object_from_security_region(region):
    gray_image = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray_image).split(':')[-1].strip()

def main(path: str):
    original = load_cv2_img(path)
    if DEBUG:
        show_img('Image', image)

    bx, by, bw, bh = security_check_img_boundary(original)
    image = original[by:by+bh, bx:bx+bw]

    if DEBUG:
        show_img('Image - Cropped Region', image)

    xx, yy, ww, hh = find_color_cube(image, security_item_rgb, 10)
    security_object = object_from_security_region(image[yy:yy+hh,xx:xx+ww])

    if DEBUG:
        show_img('Security Question', image[yy:yy+hh,xx:xx+ww])
    
    items_region = image[yy+hh+20:-95, 23:-23]
    
    if DEBUG:
        show_img('Region', items_region)

    msk = create_color_mask(items_region, [177,177,162])
    line_msk = create_color_mask(items_region, border_color_2_rgb, tolerance=5, conv=False)
    
    line_msk = cv2.cvtColor(line_msk, cv2.COLOR_RGB2BGR)
    line_msk = mask_straight_lines(line_msk)

    if DEBUG:
        show_img('Grid - Mask', line_msk)

    grid, occupied_cells = draw_grid(items_region, msk, line_msk)
    if DEBUG:
        show_img('Grid - Overlay', grid)

    imgs = []
    for start, end in occupied_cells:
        x,y = start
        x2,y2 = end
        imgs.append(Image.fromarray(items_region[y:y2,x:x2]))

    scores = get_image_text_similarity(imgs, security_object)
    scores_tensor = torch.tensor(scores)
    softmax_scores = F.softmax(scores_tensor, dim=0).tolist()
    print(softmax_scores)

    for soft, img in zip(softmax_scores, occupied_cells):
        x, y = img[0]
        x2, y2 = img[1]

        if soft >= max(softmax_scores) - 0.5:
            ox = 0
            oy = 95+20
            cv2.rectangle(original, (bx+x+xx+ox,  oy+by+y+yy), (ox+xx+bx+x2, oy+yy+by+y2), (255, 0, 0), 4)

    if DEBUG:
        show_img('Finished', original)
    
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./finished-"+os.path.basename(path), original) 


if __name__ == "__main__":
    import sys
    main(sys.argv[1])