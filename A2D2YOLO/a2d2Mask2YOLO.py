import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import yaml
from multiprocessing import Pool, cpu_count

def print_folder_info(a2d2_path):
    print("Folder Information:")
    for folder in os.listdir(a2d2_path):
        folder_path = os.path.join(a2d2_path, folder)
        if os.path.isdir(folder_path):
            image_folder = os.path.join(folder_path, "camera", "cam_front_center")
            if os.path.exists(image_folder):
                image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
                print(f"Folder: {folder}, Number of files: {len(image_files)}")
    print("\n")


def convert_filename(original_filename):
    parts = original_filename.split('_')
    parts[1] = 'label'
    return '_'.join(parts)

def load_color_mapping(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    color_map = {}
    class_map = {}
    for color_hex, class_name in data.items():
        color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color_map[color] = class_name
        if class_name not in class_map:
            class_map[class_name] = len(class_map)
    
    return color_map, class_map

def mask_to_segments(mask, color_map, class_map):
    segments = []
    for color, cls in color_map.items():
        class_mask = np.all(mask == color, axis=-1).astype(np.uint8) * 255
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 1:  # Filter out tiny contours
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) >= 3:  # Ensure we have at least a triangle
                    segments.append((class_map[cls], approx.squeeze()))
    return segments

def process_file(args):
    filename, folder, a2d2_path, yolo_path, color_map, class_map = args
    image_folder = os.path.join(a2d2_path, folder, "camera", "cam_front_center")
    mask_folder = os.path.join(a2d2_path, folder, "label", "cam_front_center")
    
    image_path = os.path.join(image_folder, filename)
    mask_filename = convert_filename(filename)
    mask_path = os.path.join(mask_folder, mask_filename)
    
    image = Image.open(image_path)
    mask = np.array(Image.open(mask_path))
    
    segments = mask_to_segments(mask, color_map, class_map)
    
    if segments:
        img_w, img_h = image.size
        
        # Randomly assign to train or val (80% train, 20% val)
        subset = "train" if np.random.rand() < 0.8 else "val"
        
        # Save image to YOLO dataset
        new_filename = f"{folder}_{filename}"
        image.save(os.path.join(yolo_path, "images", subset, new_filename))
        
        # Save YOLOv8-seg format annotations
        label_filename = os.path.splitext(new_filename)[0] + ".txt"
        label_path = os.path.join(yolo_path, "labels", subset, label_filename)
        with open(label_path, "w") as f:
            for class_id, poly in segments:
                # Normalize polygon points
                poly_norm = poly.astype(float)
                poly_norm[:, 0] /= img_w
                poly_norm[:, 1] /= img_h
                
                # Write in YOLOv8-seg format
                f.write(f"{class_id}")
                for px, py in poly_norm:
                    f.write(f" {px} {py}")
                f.write("\n")
        
        return True
    return False

def convert_a2d2_to_yolov8seg(a2d2_path, yolo_path, json_path, num_files=None):
    color_map, class_map = load_color_mapping(json_path)
    
    os.makedirs(os.path.join(yolo_path, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_path, "labels", "val"), exist_ok=True)
    
    args_list = []
    
    # Iterate through all folders in a2d2_path
    for folder in os.listdir(a2d2_path):
        folder_path = os.path.join(a2d2_path, folder)
        if os.path.isdir(folder_path):
            image_folder = os.path.join(folder_path, "camera", "cam_front_center")
            if os.path.exists(image_folder):
                # Get list of image files
                image_files = [f for f in os.listdir(image_folder) if f.endswith(".png")]
                
                # If num_files is specified, limit the number of files to process per folder
                if num_files is not None:
                    image_files = image_files[:num_files]
                
                # Prepare arguments for parallel processing
                args_list.extend([(filename, folder, a2d2_path, yolo_path, color_map, class_map) for filename in image_files])
    
    # Use multiprocessing to process files in parallel
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, args_list), total=len(args_list)))
    
    # Save dataset.yaml
    dataset_config = {
        'train': os.path.join(yolo_path, "images", "train"),
        'val': os.path.join(yolo_path, "images", "val"),
        'nc': len(class_map),
        'names': list(sorted(class_map.keys(), key=lambda x: class_map[x]))
    }
    
    with open(os.path.join(yolo_path, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

# Usage
a2d2_path = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic/"
yolo_path = "/media/parashuram/AutoData/yolov8_seg"
json_path = "/media/parashuram/AutoData/A2D2/camera_lidar_semantic/class_list.json"

print_folder_info(a2d2_path)

# To process all files in all folders:
convert_a2d2_to_yolov8seg(a2d2_path, yolo_path, json_path)

# To process only a specific number of files (e.g., 10) per folder:
# convert_a2d2_to_yolov8seg(a2d2_path, yolo_path, json_path, num_files=10)