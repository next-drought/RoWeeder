"""
RoWeeder: Unsupervised Weed Mapping through Crop-Row Detection
Based on the RoWeeder methodology combining crop-row detection with vegetation segmentation.

This script is a combined version of the crop row detection and main pipeline
implementation to avoid import issues.
"""

import os
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from matplotlib.colors import LinearSegmentedColormap
from skimage.segmentation import slic
from scipy.stats import ks_2samp
import math
import json
from pathlib import Path
import shutil

# Create a custom colormap for NDVI visualization
ndvi_cmap = LinearSegmentedColormap.from_list(
    'ndvi',
    [(0.0, 'red'), (0.5, 'yellow'), (1.0, 'green')],
    N=256
)

class NDVIVegetationDetector:
    """
    Detects vegetation using NDVI (Normalized Difference Vegetation Index)
    """
    def __init__(self, threshold=0.2, red_idx=0, nir_idx=3):
        self.threshold = threshold
        self.nir_idx = nir_idx
        self.red_idx = red_idx

    def calculate_ndvi(self, image):
        """
        Calculate NDVI without thresholding

        Args:
            image: Tensor with shape [C, H, W] containing NIR and Red bands

        Returns:
            NDVI tensor
        """
        ndvi = (image[self.nir_idx] - image[self.red_idx]) / (image[self.nir_idx] + image[self.red_idx] + 1e-8)
        ndvi[torch.isnan(ndvi)] = 0
        return ndvi

    def get_vegetation_mask(self, ndvi):
        """
        Create binary vegetation mask based on NDVI threshold

        Args:
            ndvi: NDVI tensor

        Returns:
            Binary mask of vegetation
        """
        return (ndvi > self.threshold).type(torch.uint8) * 255


class CropRowDetector:
    """
    Detects crop rows using Hough transform
    """
    def __init__(self, 
                 threshold=100, 
                 min_line_length=50, 
                 max_line_gap=10,
                 rotation_angle=0,
                 uniformity_pvalue=0.1,
                 filter_lines=True):
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.rotation_angle = rotation_angle  # Degrees
        self.uniformity_pvalue = uniformity_pvalue
        self.filter_lines = filter_lines
        
    def rotate_image(self, image):
        """
        Rotates an image to align crop rows with horizontal axis
        
        Args:
            image: Image to rotate (numpy array)
            
        Returns:
            Rotated image
        """
        if self.rotation_angle == 0:
            return image
            
        # Get dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, self.rotation_angle, 1.0)
        
        # Apply rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return rotated_image
        
    def detect_crop_rows(self, veg_mask, original_image=None, debug_dir=None, field_name=None):
        """
        Detect crop rows using Hough transform
        
        Args:
            veg_mask: Binary vegetation mask (numpy array)
            original_image: Original RGB image for visualization (optional)
            debug_dir: Directory to save debug images (optional)
            field_name: Field name for debug images (optional)
            
        Returns:
            Crop row mask, detected lines, angle distribution
        """
        # Ensure mask is numpy array and uint8
        if isinstance(veg_mask, torch.Tensor):
            veg_mask = veg_mask.cpu().numpy().astype(np.uint8)
        
        # Make sure mask is 2D
        if veg_mask.ndim > 2:
            veg_mask = veg_mask.squeeze()
        
        # Rotate mask to align crop rows
        rotated_mask = self.rotate_image(veg_mask)
        
        # Apply Hough Transform
        lines = cv2.HoughLinesP(
            rotated_mask, 
            rho=1, 
            theta=np.pi/180, 
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        # Create crop row mask
        crop_row_mask = np.zeros_like(rotated_mask)
        
        # For checking angle distribution uniformity
        angles = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angles.append(math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
                cv2.line(crop_row_mask, (x1, y1), (x2, y2), 255, 5)
        
        # Check uniformity of angles (if angles are uniform, likely false positives)
        uniform_distribution = False
        if self.filter_lines and len(angles) > 5:
            # Create uniform distribution over [-90, 90] for comparison
            uniform_sample = np.linspace(-90, 90, len(angles))
            # Kolmogorov-Smirnov test to check if angle distribution is uniform
            _, p_value = ks_2samp(np.array(angles), uniform_sample)
            uniform_distribution = p_value > self.uniformity_pvalue
            
            if uniform_distribution:
                print(f"Detected angle distribution is likely uniform (p-value: {p_value:.3f})")
                print(f"This suggests no clear crop rows. Discarding lines.")
                crop_row_mask = np.zeros_like(rotated_mask)
                lines = None
        
        # Debug visualization
        if debug_dir is not None and original_image is not None and field_name is not None:
            # Save the rotated mask
            plt.figure(figsize=(10, 10))
            plt.imshow(rotated_mask, cmap='gray')
            plt.title(f"Rotated Vegetation Mask (angle: {self.rotation_angle}°)")
            plt.savefig(os.path.join(debug_dir, f"{field_name}_rotated_mask.png"))
            plt.close()
            
            # Visualize detected lines
            if original_image is not None:
                # Ensure RGB format
                if isinstance(original_image, torch.Tensor):
                    rgb_image = original_image[:3].permute(1, 2, 0).cpu().numpy()
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                else:
                    rgb_image = original_image
                
                rotated_rgb = self.rotate_image(rgb_image)
                
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(rotated_rgb)
                plt.title("RGB Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(rotated_rgb)
                plt.imshow(crop_row_mask, alpha=0.5, cmap='Purples')
                plt.title(f"Detected Crop Rows (Count: {0 if lines is None else len(lines)})")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, f"{field_name}_crop_rows.png"))
                plt.close()
                
                # Plot angle distribution
                if lines is not None and len(angles) > 0:
                    plt.figure(figsize=(10, 6))
                    plt.hist(angles, bins=36, alpha=0.7)
                    plt.title(f"Angle Distribution (Uniform: {uniform_distribution})")
                    plt.xlabel("Angle (degrees)")
                    plt.ylabel("Count")
                    plt.savefig(os.path.join(debug_dir, f"{field_name}_angle_distribution.png"))
                    plt.close()
        
        return crop_row_mask, lines, angles


class SLICSegmenter:
    """
    Segments vegetation into superpixels using SLIC
    """
    def __init__(self, n_segments_factor=0.005, compactness=20, sigma=1):
        self.n_segments_factor = n_segments_factor
        self.compactness = compactness
        self.sigma = sigma
        
    def segment(self, image, mask=None):
        """
        Apply SLIC segmentation
        
        Args:
            image: RGB image (numpy array)
            mask: Optional mask to restrict segmentation (numpy array)
            
        Returns:
            Segmentation labels
        """
        # Ensure RGB format
        if isinstance(image, torch.Tensor):
            rgb_image = image[:3].permute(1, 2, 0).cpu().numpy()
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
        else:
            rgb_image = image
        
        # Downsample the image for faster processing
        scale_factor = 0.25  # Process at 25% of original size
        h, w = rgb_image.shape[:2]
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        
        if small_h > 0 and small_w > 0:  # Safety check
            # Downsample RGB image
            from skimage.transform import resize
            small_rgb = resize(rgb_image, (small_h, small_w), 
                            anti_aliasing=True, preserve_range=True).astype(np.uint8)
            
            # Calculate number of segments based on downsampled image size
            n_segments = int(self.n_segments_factor * small_h * small_w)
            # Ensure a reasonable minimum number of segments
            n_segments = max(n_segments, 100)
            
            if mask is not None:
                # Downsample mask
                small_mask = resize(mask, (small_h, small_w), 
                                anti_aliasing=False, preserve_range=True).astype(np.uint8)
                
                # Only apply SLIC on masked areas
                mask_bool = small_mask > 0
                small_segments = np.zeros((small_h, small_w), dtype=np.int32)
                
                # Only perform SLIC if mask has enough pixels
                if np.sum(mask_bool) > 100:
                    # Apply SLIC on the masked region with improved parameters
                    try:
                        small_masked_segments = slic(
                            small_rgb, 
                            n_segments=n_segments, 
                            compactness=self.compactness, 
                            sigma=self.sigma,
                            mask=mask_bool,
                            enforce_connectivity=True
                        )
                        
                        # Copy segmentation to output
                        small_segments[mask_bool] = small_masked_segments[mask_bool]
                    except Exception as e:
                        print(f"SLIC segmentation error: {e}")
                        # Fallback to simple segmentation - connected components
                        from skimage.measure import label
                        print("Falling back to connected components labeling")
                        small_segments = label(mask_bool)
            else:
                # Apply SLIC on entire image with improved parameters
                try:
                    small_segments = slic(
                        small_rgb, 
                        n_segments=n_segments, 
                        compactness=self.compactness, 
                        sigma=self.sigma,
                        max_iter=100,
                        enforce_connectivity=True,
                        min_size_factor=0.01
                    )
                except Exception as e:
                    print(f"SLIC segmentation error: {e}")
                    # Just return a blank segmentation in this case
                    small_segments = np.zeros((small_h, small_w), dtype=np.int32)
            
            # Upsample the segments back to original size
            segments = resize(small_segments, (h, w), order=0, 
                            preserve_range=True).astype(np.int32)
        else:
            # If downsampling resulted in zero size, fall back to original size
            print("Warning: Downsampling resulted in zero size. Using original size.")
            
            # Calculate number of segments based on image size
            n_segments = int(self.n_segments_factor * h * w)
            # Ensure a reasonable number (much fewer segments)
            n_segments = min(n_segments, 1000)
            
            if mask is not None:
                # Only apply SLIC on masked areas
                mask_bool = mask > 0
                segments = np.zeros((h, w), dtype=np.int32)
                
                # Only perform SLIC if mask has enough pixels
                if np.sum(mask_bool) > 100:
                    try:
                        # Apply SLIC with more conservative parameters
                        masked_segments = slic(
                            rgb_image, 
                            n_segments=n_segments, 
                            compactness=30,  # Higher compactness
                            sigma=2,         # Higher sigma
                            mask=mask_bool,
                            max_iter=50,     # Fewer iterations
                            enforce_connectivity=True,
                            min_size_factor=0.02
                        )
                        
                        # Copy segmentation to output
                        segments[mask_bool] = masked_segments[mask_bool]
                    except Exception as e:
                        print(f"SLIC segmentation error: {e}")
                        # Fallback to connected components
                        from skimage.measure import label
                        print("Falling back to connected components labeling")
                        segments = label(mask_bool)
            else:
                try:
                    # Apply SLIC with more conservative parameters
                    segments = slic(
                        rgb_image, 
                        n_segments=n_segments, 
                        compactness=30,
                        sigma=2,
                        max_iter=50,
                        enforce_connectivity=True,
                        min_size_factor=0.02
                    )
                except Exception as e:
                    print(f"SLIC segmentation error: {e}")
                    segments = np.zeros((h, w), dtype=np.int32)
        
        return segments

def classify_plants(segments, crop_row_mask, veg_mask):
    """
    Classify plant segments as crops or weeds based on overlap with crop rows
    
    Args:
        segments: Segmentation labels from SLIC
        crop_row_mask: Binary mask of crop rows
        veg_mask: Binary vegetation mask
        
    Returns:
        Classification map (0=background, 1=crop, 2=weed), plant objects
    """
    # Initialize classification map
    classification = np.zeros_like(segments, dtype=np.uint8)
    
    # Extract unique segment labels (excluding 0 background)
    segment_labels = np.unique(segments)
    
    # For storing plant objects
    plant_objects = []
    
    # Process each segment
    for label in segment_labels:
        # Create mask for current segment
        segment_mask = (segments == label).astype(np.uint8)
        
        # Check if segment corresponds to vegetation
        plant_mask = segment_mask & (veg_mask > 0)
        plant_area = np.sum(plant_mask)
        
        # If segment has vegetation
        if plant_area > 0:
            # Check overlap with crop rows
            crop_overlap = np.sum(plant_mask & (crop_row_mask > 0))
            
            # Classify segment
            if crop_overlap > 0:
                # Plant overlaps with crop row -> crop
                classification[segments == label] = 1
                class_name = 'crop'
            else:
                # Plant doesn't overlap with crop row -> weed
                classification[segments == label] = 2
                class_name = 'weed'
                
            # Get plant contour
            plant_mask_uint8 = plant_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(plant_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add to plant objects
                plant_objects.append({
                    'contour': largest_contour,
                    'class': class_name,
                    'bbox': (x, y, w, h),
                    'area': plant_area,
                    'centroid': (x + w//2, y + h//2)
                })
    
    return classification, plant_objects


def visualize_classification(image, classification, plant_objects, output_path):
    """
    Visualize plant classification and save the visualization
    
    Args:
        image: Original image
        classification: Classification map
        plant_objects: List of plant objects
        output_path: Path to save visualization
    """
    # Ensure RGB format
    if isinstance(image, torch.Tensor):
        rgb_image = image[:3].permute(1, 2, 0).cpu().numpy()
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
    else:
        rgb_image = image
    
    # Create color maps for classification
    cmap = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
    # Class 1 = Crop (Green)
    cmap[classification == 1] = [0, 255, 0]
    # Class 2 = Weed (Red)
    cmap[classification == 2] = [255, 0, 0]
    
    # Create visualization figure
    plt.figure(figsize=(20, 15))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot classification
    plt.subplot(2, 2, 2)
    plt.imshow(rgb_image)
    plt.imshow(cmap, alpha=0.5)
    plt.title(f'Classification (Green: Crop, Red: Weed)')
    plt.axis('off')
    
    # Plot plant objects
    plt.subplot(2, 2, 3)
    plt.imshow(rgb_image)
    
    # Draw plant contours and bounding boxes
    contour_img = rgb_image.copy()
    for plant in plant_objects:
        color = (0, 255, 0) if plant['class'] == 'crop' else (255, 0, 0)
        cv2.drawContours(contour_img, [plant['contour']], -1, color, 2)
        x, y, w, h = plant['bbox']
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), color, 1)
    
    plt.imshow(contour_img)
    plt.title(f'Plant Objects (Count: {len(plant_objects)})')
    plt.axis('off')
    
    # Plot classification overlay
    plt.subplot(2, 2, 4)
    # Black background
    plt.imshow(np.zeros_like(rgb_image))
    # Green for crops
    crop_mask = (classification == 1).astype(np.uint8) * 255
    crop_mask_rgb = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
    crop_mask_rgb[:, :, 1] = crop_mask  # Green channel
    plt.imshow(crop_mask_rgb, alpha=1.0)
    # Red for weeds
    weed_mask = (classification == 2).astype(np.uint8) * 255
    weed_mask_rgb = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
    weed_mask_rgb[:, :, 0] = weed_mask  # Red channel
    plt.imshow(weed_mask_rgb, alpha=1.0)
    
    plt.title('Classification Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # Create mask for YOLO training
    crops_mask = (classification == 1).astype(np.uint8) * 255
    weeds_mask = (classification == 2).astype(np.uint8) * 255
    
    # Save masks
    crops_mask_path = os.path.splitext(output_path)[0] + '_crops_mask.png'
    weeds_mask_path = os.path.splitext(output_path)[0] + '_weeds_mask.png'
    classification_rgb_path = os.path.splitext(output_path)[0] + '_classification.png'
    
    cv2.imwrite(crops_mask_path, crops_mask)
    cv2.imwrite(weeds_mask_path, weeds_mask)
    
    # Save color classification for easy viewing
    classification_rgb = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 3), dtype=np.uint8)
    classification_rgb[classification == 1] = [0, 255, 0]
    classification_rgb[classification == 2] = [255, 0, 0]
    cv2.imwrite(classification_rgb_path, cv2.cvtColor(classification_rgb, cv2.COLOR_RGB2BGR))
    
    # Save RGB image for reference
    rgb_path = os.path.splitext(output_path)[0] + '_rgb.png'
    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    return rgb_path, crops_mask_path, weeds_mask_path


def prepare_yolo_dataset(rgb_images, plant_objects_list, output_dir, split_ratio=0.8):
    """
    Prepare dataset for YOLO training with individual plants
    
    Args:
        rgb_images: List of RGB image paths
        plant_objects_list: List of plant objects for each image
        output_dir: Output directory for YOLO dataset
        split_ratio: Train/val split ratio
    """
    # Create directory structure
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)

    # Create dataset.yaml
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml_str = f"path: {output_dir}\n"
        yaml_str += f"train: images/train\n"
        yaml_str += f"val: images/val\n"
        yaml_str += f"names:\n"
        yaml_str += f"  0: crop\n"
        yaml_str += f"  1: weed\n"
        f.write(yaml_str)

    # Split data
    num_train = int(len(rgb_images) * split_ratio)
    train_indices = np.random.choice(len(rgb_images), num_train, replace=False)
    val_indices = np.array([i for i in range(len(rgb_images)) if i not in train_indices])

    # Process train images
    for i in train_indices:
        # Copy image
        img_filename = os.path.basename(rgb_images[i])
        shutil.copy(rgb_images[i], os.path.join(output_dir, 'images', 'train', img_filename))

        # Get image dimensions
        img = cv2.imread(rgb_images[i])
        img_height, img_width = img.shape[:2]

        # Create label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        with open(os.path.join(output_dir, 'labels', 'train', label_filename), 'w') as f:
            for plant in plant_objects_list[i]:
                x, y, w, h = plant['bbox']
                
                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Class ID: 0 for crop, 1 for weed
                class_id = 0 if plant['class'] == 'crop' else 1
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Process val images
    for i in val_indices:
        # Copy image
        img_filename = os.path.basename(rgb_images[i])
        shutil.copy(rgb_images[i], os.path.join(output_dir, 'images', 'val', img_filename))

        # Get image dimensions
        img = cv2.imread(rgb_images[i])
        img_height, img_width = img.shape[:2]

        # Create label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        with open(os.path.join(output_dir, 'labels', 'val', label_filename), 'w') as f:
            for plant in plant_objects_list[i]:
                x, y, w, h = plant['bbox']
                
                # Convert to YOLO format (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Class ID: 0 for crop, 1 for weed
                class_id = 0 if plant['class'] == 'crop' else 1
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print(f"YOLO dataset prepared in {output_dir}")
    print(f"Training images: {len(train_indices)}")
    print(f"Validation images: {len(val_indices)}")


def load_multispectral_image(image_path, format='tif', channels=None):
    """
    Load a multispectral image from a directory structure or a single file

    Args:
        image_path: Path to the image file or directory
        format: Image format ('tif' or 'png')
        channels: List of channel names to load (default: R, G, B, NIR)

    Returns:
        Tensor with shape [C, H, W]
    """
    if channels is None:
        channels = ['R', 'G', 'B', 'NIR']

    if os.path.isdir(image_path):
        # Handle RedEdge dataset structure
        if format == 'tif':
            # For TIF format, load from reflectance-tif directory
            channel_tensors = []
            channel_files = {
                'R': 'transparent_reflectance_red.tif',
                'G': 'transparent_reflectance_green.tif',
                'B': 'transparent_reflectance_blue.tif',
                'NIR': 'transparent_reflectance_nir.tif',
                'RE': 'transparent_reflectance_rededge.tif'
            }

            for channel in channels:
                if channel not in channel_files:
                    raise ValueError(f"Channel {channel} not supported for TIF format")

                channel_path = os.path.join(image_path, channel_files[channel])

                if not os.path.exists(channel_path):
                    raise ValueError(f"Channel file {channel_path} not found")

                # Use PIL to open TIFF
                img = Image.open(channel_path)
                channel_array = np.array(img).astype(np.float32)

                # Handle single band images
                if channel_array.ndim == 2:
                    channel_tensor = torch.from_numpy(channel_array).float().unsqueeze(0)
                else:
                    # If the array is 3D with channels last, transpose to channels first
                    channel_array = np.transpose(channel_array, (2, 0, 1))
                    channel_tensor = torch.from_numpy(channel_array).float()

                channel_tensors.append(channel_tensor)

            # Stack channels
            image_tensor = torch.cat([t for t in channel_tensors])

        else:  # PNG format
            # For PNG format, load from composite-png directory
            channel_tensors = []

            for channel in channels:
                channel_path = os.path.join(image_path, f"{channel}.png")

                if not os.path.exists(channel_path):
                    raise ValueError(f"Channel file {channel_path} not found")

                # Use PIL instead of torchvision to avoid libjpeg issues
                img = Image.open(channel_path)
                channel_array = np.array(img).astype(np.float32)
                if channel_array.ndim == 3:
                    channel_array = channel_array[:,:,0]  # Take first channel if RGB
                channel_tensor = torch.from_numpy(channel_array).float().unsqueeze(0)

                channel_tensors.append(channel_tensor)

            # Stack channels
            image_tensor = torch.cat(channel_tensors)

    else:
        # Single file (e.g., TIFF with multiple bands)
        if image_path.lower().endswith(('.tif', '.tiff')):
            # Use PIL to open TIFF
            img = Image.open(image_path)

            # Check if it's a multiband image
            if img.mode == 'I':  # Single band
                image_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0)
            else:
                # Convert to numpy array and then to tensor
                img_array = np.array(img)

                # If the array is 3D with channels last, transpose to channels first
                if img_array.ndim == 3:
                    img_array = np.transpose(img_array, (2, 0, 1))

                image_tensor = torch.from_numpy(img_array).float()
        else:
            # Regular image file - use PIL instead of torchvision
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # If the array is 3D with channels last, transpose to channels first
            if img_array.ndim == 3:
                img_array = np.transpose(img_array, (2, 0, 1))
                
            image_tensor = torch.from_numpy(img_array).float()

    # Normalize to [0, 1] range if needed
    if image_tensor.max() > 1.0:
        image_tensor /= 255.0

    return image_tensor


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Crop Row Detection and Plant Identification')
    parser.add_argument('--dataset', type=str, default='~/Dropbox/DEV/Data/RedEdge',
                        help='Path to the RedEdge dataset root directory')
    parser.add_argument('--fields', type=str, nargs='+', default=['000', '001', '002', '003', '004'],
                        help='Field IDs to process')
    parser.add_argument('--format', type=str, choices=['tif', 'png'], default='tif',
                        help='Image format to use (tif or png)')
    parser.add_argument('--ndvi-threshold', type=float, default=0.2,
                        help='NDVI threshold for vegetation detection')
    parser.add_argument('--hough-threshold', type=int, default=100, 
                        help='Threshold for Hough transform')
    parser.add_argument('--min-line-length', type=int, default=50,
                        help='Minimum line length for Hough transform')
    parser.add_argument('--max-line-gap', type=int, default=10,
                        help='Maximum line gap for Hough transform')
    parser.add_argument('--rotation-angle', type=float, default=0,
                        help='Rotation angle to align crop rows (degrees)')
    parser.add_argument('--min-area', type=int, default=10,
                        help='Minimum contour area to keep')
    parser.add_argument('--n-segments-factor', type=float, default=0.005,
                        help='Factor to calculate number of segments for SLIC')
    parser.add_argument('--red-idx', type=int, default=2,
                        help='Index of the red band in the image')
    parser.add_argument('--nir-idx', type=int, default=3,
                        help='Index of the NIR band in the image')
    parser.add_argument('--output-dir', type=str, default='crop_detection_results',
                        help='Directory to save results')
    parser.add_argument('--prepare-yolo', action='store_true',
                        help='Prepare dataset for YOLO training')
    parser.add_argument('--yolo-dir', type=str, default='yolo_dataset',
                        help='Directory to save YOLO dataset')
    parser.add_argument('--debug', action='store_true',
                        help='Save intermediate debug images')
                        
    args = parser.parse_args()
    
    # Expand user directory if needed
    dataset_path = os.path.expanduser(args.dataset)
    print(f"Using dataset path: {dataset_path}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create debug directory if needed
    debug_dir = os.path.join(args.output_dir, 'debug')
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Create NDVI detector
    ndvi_detector = NDVIVegetationDetector(
        threshold=args.ndvi_threshold,
        red_idx=args.red_idx,
        nir_idx=args.nir_idx
    )
    
    # Create crop row detector
    crop_row_detector = CropRowDetector(
        threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        rotation_angle=args.rotation_angle
    )
    
    # Create SLIC segmenter
    # slic_segmenter = SLICSegmenter(
    #     n_segments_factor=args.n_segments_factor,
    #     compactness=20,
    #     sigma=1
    # )
    # Create SLIC segmenter with more robust parameters
    slic_segmenter = SLICSegmenter(
        n_segments_factor=0.0001,  # Much smaller value
        compactness=30,            # Increased from 20
        sigma=1
    )

    # Process each field
    rgb_images = []
    all_plant_objects = []

    for field in args.fields:
        field_path = os.path.join(dataset_path, field)
        
        if not os.path.exists(field_path):
            print(f"Warning: Field {field} not found at {field_path}")
            continue
            
        print(f"Processing field {field}...")
        
        # Set up paths based on format choice
        if args.format == 'tif':
            # For TIF format, use the reflectance-tif directory
            image_path = os.path.join(field_path, 'reflectance-tif')
        else:
            # For PNG format, use the composite-png directory
            image_path = os.path.join(field_path, 'composite-png')
            
        print(f"Checking if path exists: {image_path}")
        if not os.path.exists(image_path):
            print(f"Error: Path {image_path} does not exist")
            continue
            
        # List files in directory for debugging
        print(f"Files in {image_path}:")
        for f in os.listdir(image_path):
            print(f"  {f}")
            
        # Load image
        print(f"Loading image from {image_path}...")
        try:
            image = load_multispectral_image(image_path, format=args.format)
            print(f"Image loaded with shape: {image.shape}")
            
            # Save a sample of the loaded image for debugging
            if args.debug:
                rgb_debug = image[:3].permute(1, 2, 0).cpu().numpy()
                if rgb_debug.max() <= 1.0:
                    rgb_debug = (rgb_debug * 255).astype(np.uint8)
                plt.figure(figsize=(10, 10))
                plt.imshow(rgb_debug)
                plt.title(f"RGB Channels (0,1,2) from loaded image")
                plt.savefig(os.path.join(debug_dir, f"field_{field}_rgb_debug.png"))
                plt.close()
                
                # Save individual bands for debugging
                for i, band_name in enumerate(['Band 0', 'Band 1', 'Band 2', 'Band 3', 'Band 4']):
                    if i < image.shape[0]:
                        plt.figure(figsize=(10, 10))
                        plt.imshow(image[i].cpu().numpy(), cmap='gray')
                        plt.title(f"{band_name} (Index {i})")
                        plt.colorbar()
                        plt.savefig(os.path.join(debug_dir, f"field_{field}_band_{i}.png"))
                        plt.close()
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Calculate NDVI
        print(f"Calculating NDVI using Red index: {args.red_idx}, NIR index: {args.nir_idx}...")
        try:
            ndvi = ndvi_detector.calculate_ndvi(image)
            print(f"NDVI calculated. Min: {ndvi.min().item():.3f}, Max: {ndvi.max().item():.3f}, Mean: {ndvi.mean().item():.3f}")
            
            # Save NDVI for debugging
            if args.debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(ndvi.cpu().numpy(), cmap=ndvi_cmap, vmin=-1, vmax=1)
                plt.colorbar(label='NDVI')
                plt.title(f"NDVI (Min: {ndvi.min().item():.3f}, Max: {ndvi.max().item():.3f}, Mean: {ndvi.mean().item():.3f})")
                plt.savefig(os.path.join(debug_dir, f"field_{field}_ndvi.png"))
                plt.close()
        except Exception as e:
            print(f"Error calculating NDVI: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Get vegetation mask
        print(f"Creating vegetation mask with threshold: {args.ndvi_threshold}...")
        try:
            veg_mask = ndvi_detector.get_vegetation_mask(ndvi)
            veg_pixels = (veg_mask > 0).sum().item()
            total_pixels = veg_mask.numel()
            veg_percentage = (veg_pixels / total_pixels) * 100
            print(f"Vegetation mask created. Vegetation pixels: {veg_pixels} ({veg_percentage:.2f}% of image)")
            
            # Save vegetation mask for debugging
            if args.debug:
                plt.figure(figsize=(10, 10))
                plt.imshow(veg_mask.squeeze().cpu().numpy(), cmap='gray')
                plt.title(f"Vegetation Mask (Threshold: {args.ndvi_threshold}, {veg_percentage:.2f}% vegetation)")
                plt.savefig(os.path.join(debug_dir, f"field_{field}_veg_mask.png"))
                plt.close()
        except Exception as e:
            print(f"Error creating vegetation mask: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # If no vegetation is detected, try with a lower threshold
        if veg_pixels < 100:
            print("Very few vegetation pixels detected. Trying with a lower threshold...")
            for test_threshold in [0.15, 0.1, 0.05]:
                print(f"Testing threshold: {test_threshold}")
                test_ndvi = ndvi_detector.calculate_ndvi(image)
                test_mask = (test_ndvi > test_threshold).type(torch.uint8) * 255
                test_pixels = (test_mask > 0).sum().item()
                test_percentage = (test_pixels / total_pixels) * 100
                print(f"  Found {test_pixels} vegetation pixels ({test_percentage:.2f}%) with threshold={test_threshold}")
                
                if test_pixels > 100:
                    print(f"Using threshold={test_threshold} instead")
                    veg_mask = test_mask
                    break
                    
        # Detect crop rows
        print(f"Detecting crop rows...")
        try:
            # Convert vegetation mask to numpy array
            veg_mask_np = veg_mask.squeeze().cpu().numpy() if isinstance(veg_mask, torch.Tensor) else veg_mask.squeeze()
            
            # Get RGB image for visualization
            rgb_image = image[:3].permute(1, 2, 0).cpu().numpy()
            if rgb_image.max() <= 1.0:
                rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Detect crop rows
            crop_row_mask, lines, angles = crop_row_detector.detect_crop_rows(
                veg_mask_np, 
                original_image=rgb_image,
                debug_dir=debug_dir if args.debug else None,
                field_name=f"field_{field}"
            )
            
            if lines is None or len(lines) == 0:
                print("No crop rows detected. Adjusting parameters and trying again...")
                
                # Try with different thresholds
                for threshold in [80, 60, 40]:
                    print(f"  Trying with Hough threshold: {threshold}")
                    crop_row_detector.threshold = threshold
                    crop_row_mask, lines, angles = crop_row_detector.detect_crop_rows(
                        veg_mask_np, 
                        original_image=rgb_image,
                        debug_dir=debug_dir if args.debug else None,
                        field_name=f"field_{field}_threshold_{threshold}"
                    )
                    
                    if lines is not None and len(lines) > 0:
                        print(f"  Success! Detected {len(lines)} lines with threshold={threshold}")
                        break
                
                # Try with different minimum line length
                if lines is None or len(lines) == 0:
                    for min_length in [40, 30, 20]:
                        print(f"  Trying with min line length: {min_length}")
                        crop_row_detector.min_line_length = min_length
                        crop_row_mask, lines, angles = crop_row_detector.detect_crop_rows(
                            veg_mask_np, 
                            original_image=rgb_image,
                            debug_dir=debug_dir if args.debug else None,
                            field_name=f"field_{field}_minlength_{min_length}"
                        )
                        
                        if lines is not None and len(lines) > 0:
                            print(f"  Success! Detected {len(lines)} lines with min_length={min_length}")
                            break
            
            # If still no crop rows, create artificial ones based on manual spacing
            if lines is None or len(lines) == 0:
                print("Still no crop rows detected. Creating artificial rows based on typical spacing...")
                height, width = veg_mask_np.shape[:2]
                crop_row_mask = np.zeros((height, width), dtype=np.uint8)
                
                # Try to find typical row spacing by looking at peaks in vegetation patterns
                rows_y = []
                for y in range(0, height):
                    row_density = np.sum(veg_mask_np[y, :]) / width
                    if row_density > 0.1:  # If more than 10% of the row is vegetation
                        rows_y.append(y)
                
                if len(rows_y) > 0:
                    # Create artificial crop rows
                    row_thickness = 10
                    for y in rows_y:
                        crop_row_mask[max(0, y-row_thickness//2):min(height, y+row_thickness//2), :] = 255
                    
                    # Save artificial crop rows for debugging
                    if args.debug:
                        plt.figure(figsize=(15, 10))
                        plt.imshow(rgb_image)
                        plt.imshow(crop_row_mask, alpha=0.5, cmap='Purples')
                        plt.title(f"Artificial Crop Rows (Count: {len(rows_y)})")
                        plt.axis('off')
                        plt.savefig(os.path.join(debug_dir, f"field_{field}_artificial_rows.png"))
                        plt.close()
                else:
                    print("Could not detect any vegetation patterns for artificial rows. Skipping crop row detection.")
        except Exception as e:
            print(f"Error detecting crop rows: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Apply SLIC segmentation
        print(f"Applying SLIC segmentation...")
        try:
            # Apply SLIC to get plant segments
            segments = slic_segmenter.segment(rgb_image, veg_mask_np)
            
            # Debug visualization of segments
            if args.debug:
                # Create random colors for segments
                from skimage.color import label2rgb
                segment_img = label2rgb(segments, rgb_image, alpha=0.5)
                
                plt.figure(figsize=(15, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(rgb_image)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(segment_img)
                plt.title(f"SLIC Segments")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, f"field_{field}_slic_segments.png"))
                plt.close()
        except Exception as e:
            print(f"Error applying SLIC segmentation: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        # Classify plants as crops or weeds
        print(f"Classifying plants...")
        try:
            classification, plant_objects = classify_plants(segments, crop_row_mask, veg_mask_np)
            
            print(f"Found {len(plant_objects)} plant objects")
            # Count crops and weeds
            num_crops = sum(1 for p in plant_objects if p['class'] == 'crop')
            num_weeds = sum(1 for p in plant_objects if p['class'] == 'weed')
            print(f"  Crops: {num_crops}")
            print(f"  Weeds: {num_weeds}")
            
            # Visualize classification
            output_path = os.path.join(args.output_dir, f"field_{field}_classification.png")
            rgb_path, crops_mask_path, weeds_mask_path = visualize_classification(
                rgb_image, 
                classification, 
                plant_objects, 
                output_path
            )
            
            # Add to list for YOLO dataset preparation
            if args.prepare_yolo:
                rgb_images.append(rgb_path)
                all_plant_objects.append(plant_objects)
                
        except Exception as e:
            print(f"Error classifying plants: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    # Prepare YOLO dataset
    if args.prepare_yolo and rgb_images:
        print("\nPreparing YOLO dataset...")
        prepare_yolo_dataset(rgb_images, all_plant_objects, args.yolo_dir)
        
        print("\nYOLO dataset preparation complete!")
        print(f"To train YOLOv8 on this dataset, run:")
        print(f"yolo task=detect mode=train data={os.path.join(args.yolo_dir, 'dataset.yaml')} epochs=100")
        
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()