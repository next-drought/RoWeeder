import os
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
from torchvision.transforms import functional as F
import argparse
from matplotlib.colors import LinearSegmentedColormap

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

    def __call__(self, image=None, ndvi=None):
        """
        Calculate vegetation mask based on NDVI threshold
        
        Args:
            image: Tensor with shape [C, H, W] containing NIR and Red bands
            ndvi: Pre-calculated NDVI (optional)
            
        Returns:
            Binary mask of vegetation
        """
        if ndvi is not None:
            ndvi_tensor = ndvi
        else:
            # Calculate NDVI
            ndvi_tensor = (image[self.nir_idx] - image[self.red_idx]) / (image[self.nir_idx] + image[self.red_idx] + 1e-8)
            # Replace NaN values with 0
            ndvi_tensor[torch.isnan(ndvi_tensor)] = 0
            
        # Create binary mask based on threshold
        veg_mask = (ndvi_tensor > self.threshold).type(torch.uint8) * 255
        return veg_mask.unsqueeze(0)

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


def calculate_connectivity(input_img):
    """
    Find connected components in a binary image
    
    Args:
        input_img: Binary image tensor or numpy array
        
    Returns:
        components: Connected components labeled image
        regions: Information about each component
    """
    # Convert to numpy if tensor
    if isinstance(input_img, torch.Tensor):
        input_img = input_img.cpu().numpy().astype(np.uint8)
    
    # Find connected components
    num_labels, components = cv2.connectedComponents(input_img)
    
    # Extract region information
    regions = []
    for label in range(1, num_labels):  # Skip background (0)
        # Find pixels belonging to this component
        y_coords, x_coords = np.where(components == label)
        
        if len(y_coords) == 0:
            continue
            
        # Calculate bounding box
        x0, y0 = np.min(x_coords), np.min(y_coords)
        x1, y1 = np.max(x_coords), np.max(y_coords)
        
        # Calculate centroid
        cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
        
        # Store region information
        regions.append({
            'label': label,
            'centroid': (cx, cy),
            'bbox': (x0, y0, x1, y1),
            'width': x1 - x0 + 1,
            'height': y1 - y0 + 1,
            'area': len(x_coords)
        })
    
    return components, regions


def extract_plants(image, plant_mask):
    """
    Extract individual plants from an image based on connected components
    
    Args:
        image: Image tensor with shape [C, H, W]
        plant_mask: Binary mask of plants
        
    Returns:
        plants: List of individual plant images
        plant_masks: List of individual plant masks
    """
    # Find connected components
    components, regions = calculate_connectivity(plant_mask)
    
    plants = []
    plant_masks = []
    plant_regions = []
    
    # Extract each plant
    for region in regions:
        label = region['label']
        x0, y0, x1, y1 = region['bbox']
        
        # Create mask for this plant
        plant_component_mask = (components == label).astype(np.uint8)
        
        # Skip very small components (likely noise)
        if region['area'] < 50:
            continue
            
        # Extract plant image
        if isinstance(image, torch.Tensor):
            # For each channel, apply the mask and crop
            plant_img = image[:, y0:y1+1, x0:x1+1].clone()
            mask_crop = torch.from_numpy(plant_component_mask[y0:y1+1, x0:x1+1])
            
            # Apply mask to each channel
            for c in range(image.shape[0]):
                plant_img[c] = plant_img[c] * mask_crop
        else:
            # For numpy array
            plant_img = image[:, y0:y1+1, x0:x1+1].copy()
            mask_crop = plant_component_mask[y0:y1+1, x0:x1+1]
            
            # Apply mask to each channel
            for c in range(image.shape[0]):
                plant_img[c] = plant_img[c] * mask_crop
        
        plants.append(plant_img)
        plant_masks.append(mask_crop)
        plant_regions.append(region)
    
    return plants, plant_masks, plant_regions


def calculate_plant_ndvi_stats(plants, plant_masks, ndvi_detector):
    """
    Calculate NDVI statistics for each plant
    
    Args:
        plants: List of plant images
        plant_masks: List of plant masks
        ndvi_detector: NDVIVegetationDetector instance
        
    Returns:
        List of dictionaries with NDVI statistics for each plant
    """
    plant_stats = []
    
    for i, (plant, mask) in enumerate(zip(plants, plant_masks)):
        # Calculate NDVI for this plant
        ndvi = ndvi_detector.calculate_ndvi(plant)
        
        # Convert mask to tensor if needed
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # Apply mask to NDVI
        masked_ndvi = ndvi * mask
        
        # Calculate statistics for non-zero values
        valid_values = masked_ndvi[mask > 0]
        
        if len(valid_values) > 0:
            mean_ndvi = valid_values.mean().item()
            min_ndvi = valid_values.min().item()
            max_ndvi = valid_values.max().item()
            median_ndvi = torch.median(valid_values).item()
        else:
            mean_ndvi = min_ndvi = max_ndvi = median_ndvi = 0
        
        # Categorize plant health based on mean NDVI
        if mean_ndvi < 0.2:
            health = "Poor"
        elif mean_ndvi < 0.4:
            health = "Fair"
        elif mean_ndvi < 0.6:
            health = "Good"
        else:
            health = "Excellent"
        
        plant_stats.append({
            'plant_id': i + 1,
            'mean_ndvi': mean_ndvi,
            'min_ndvi': min_ndvi,
            'max_ndvi': max_ndvi,
            'median_ndvi': median_ndvi,
            'health': health
        })
    
    return plant_stats


def load_multispectral_image(image_path, channels=None):
    """
    Load a multispectral image from a directory structure or a single file
    
    Args:
        image_path: Path to the image file or directory
        channels: List of channel names to load (default: R, G, B, NIR)
        
    Returns:
        Tensor with shape [C, H, W]
    """
    if channels is None:
        channels = ['R', 'G', 'B', 'NIR']
    
    if os.path.isdir(image_path):
        # Directory structure with separate files for each channel
        channel_tensors = []
        
        for channel in channels:
            # Look for files with channel name in the directory
            channel_files = [f for f in os.listdir(image_path) if channel in f]
            
            if not channel_files:
                raise ValueError(f"Channel {channel} not found in {image_path}")
                
            channel_path = os.path.join(image_path, channel_files[0])
            channel_tensor = torchvision.io.read_image(channel_path).float()
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
            # Regular image file
            image_tensor = torchvision.io.read_image(image_path).float()
    
    # Normalize to [0, 1] range
    image_tensor /= 255.0
    
    return image_tensor


def visualize_results(image, ndvi, veg_mask, components, regions, plant_stats):
    """
    Visualize the results of crop counting and health assessment
    
    Args:
        image: Original image tensor
        ndvi: NDVI tensor
        veg_mask: Vegetation mask
        components: Connected components labeled image
        regions: Information about each component
        plant_stats: NDVI statistics for each plant
    """
    # Convert tensors to numpy if needed
    if isinstance(image, torch.Tensor):
        # Use first 3 channels (RGB) for visualization
        rgb_image = image[:3].permute(1, 2, 0).cpu().numpy()
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
    else:
        rgb_image = image[:3].transpose(1, 2, 0)
        
    if isinstance(ndvi, torch.Tensor):
        ndvi_np = ndvi.cpu().numpy()
    else:
        ndvi_np = ndvi
        
    if isinstance(veg_mask, torch.Tensor):
        veg_mask_np = veg_mask.squeeze().cpu().numpy()
    else:
        veg_mask_np = veg_mask
    
    # Create figure
    plt.figure(figsize=(20, 15))
    
    # Plot original image
    plt.subplot(2, 2, 1)
    plt.imshow(rgb_image)
    plt.title('Original Image (RGB)')
    plt.axis('off')
    
    # Plot NDVI
    plt.subplot(2, 2, 2)
    plt.imshow(ndvi_np, cmap=ndvi_cmap, vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI')
    plt.axis('off')
    
    # Plot vegetation mask
    plt.subplot(2, 2, 3)
    plt.imshow(veg_mask_np, cmap='gray')
    plt.title('Vegetation Mask')
    plt.axis('off')
    
    # Plot connected components with labels
    plt.subplot(2, 2, 4)
    plt.imshow(rgb_image)
    
    # Add bounding boxes and labels
    for i, region in enumerate(regions):
        x0, y0, x1, y1 = region['bbox']
        cx, cy = region['centroid']
        
        # Get corresponding plant stats
        stats = plant_stats[i]
        
        # Draw bounding box
        rect = plt.Rectangle((x0, y0), x1-x0, y1-y0, 
                            fill=False, 
                            edgecolor='r', 
                            linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label with ID and health
        plt.text(cx, y0-5, f"#{stats['plant_id']} ({stats['health']})", 
                color='white', fontsize=8, 
                bbox=dict(facecolor='red', alpha=0.5))
    
    plt.title(f'Crop Count: {len(regions)}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('crop_analysis_results.png', dpi=300)
    plt.show()
    
    # Print plant statistics
    print(f"\nTotal plants detected: {len(plant_stats)}")
    print("\nPlant Health Statistics:")
    print("-----------------------")
    
    # Count plants by health category
    health_counts = {"Poor": 0, "Fair": 0, "Good": 0, "Excellent": 0}
    for stat in plant_stats:
        health_counts[stat['health']] += 1
    
    for health, count in health_counts.items():
        print(f"{health}: {count} plants ({count/len(plant_stats)*100:.1f}%)")
    
    print("\nDetailed Plant Information:")
    print("-------------------------")
    for stat in plant_stats:
        print(f"Plant #{stat['plant_id']}: Mean NDVI = {stat['mean_ndvi']:.3f}, Health: {stat['health']}")


def main():
    parser = argparse.ArgumentParser(description='Crop Counting and Health Assessment')
    parser.add_argument('--image', type=str, required=True, help='Path to the multispectral image or directory')
    parser.add_argument('--threshold', type=float, default=0.2, help='NDVI threshold for vegetation detection')
    parser.add_argument('--red-idx', type=int, default=0, help='Index of the red band in the image')
    parser.add_argument('--nir-idx', type=int, default=3, help='Index of the NIR band in the image')
    
    args = parser.parse_args()
    
    # Load image
    print(f"Loading image from {args.image}...")
    image = load_multispectral_image(args.image)
    print(f"Image loaded with shape: {image.shape}")
    
    # Create NDVI detector
    ndvi_detector = NDVIVegetationDetector(
        threshold=args.threshold,
        red_idx=args.red_idx,
        nir_idx=args.nir_idx
    )
    
    # Calculate NDVI
    print("Calculating NDVI...")
    ndvi = ndvi_detector.calculate_ndvi(image)
    
    # Detect vegetation
    print("Detecting vegetation...")
    veg_mask = ndvi_detector(ndvi=ndvi)
    
    # Find connected components (individual plants)
    print("Finding individual plants...")
    components, regions = calculate_connectivity(veg_mask.squeeze().numpy())
    
    # Extract individual plants
    print("Extracting individual plants...")
    plants, plant_masks, plant_regions = extract_plants(image, veg_mask.squeeze().numpy())
    
    # Calculate NDVI statistics for each plant
    print("Calculating plant health statistics...")
    plant_stats = calculate_plant_ndvi_stats(plants, plant_masks, ndvi_detector)
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(image, ndvi, veg_mask, components, plant_regions, plant_stats)
    
    print(f"\nAnalysis complete! Results saved to 'crop_analysis_results.png'")


if __name__ == "__main__":
    main()
