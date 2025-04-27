# Crop Counter and Health Assessment Tool for RedEdge Imagery

This tool analyzes multispectral drone imagery to count individual crops and assess their health using NDVI (Normalized Difference Vegetation Index). It's specifically adapted for the RedEdge dataset structure.

## Features

- **Crop Counting**: Identifies and counts individual plants using connected component analysis
- **Health Assessment**: Calculates NDVI for each plant and categorizes health status
- **Visualization**: Generates visual output with plant identification and health status

## Dataset Structure

The tool is designed to work with the RedEdge dataset structure:

```
/path/to/RedEdge/
├── 000, 001, 002, 003, 004 (field folders)
│   ├── composite-png
│   │   ├── B.png
│   │   ├── G.png
│   │   ├── NIR.png
│   │   ├── R.png
│   │   └── ...
│   ├── groundtruth
│   └── reflectance-tif
│       ├── transparent_reflectance_blue.tif
│       ├── transparent_reflectance_green.tif
│       ├── transparent_reflectance_nir.tif
│       ├── transparent_reflectance_red.tif
│       └── transparent_reflectance_rededge.tif
```

## Installation

### Using Conda (Recommended)

```bash
# Create and activate the conda environment
conda env create -f crop_counter_environment.yml
conda activate crop-counter
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n crop-counter python=3.11
conda activate crop-counter

# Install dependencies
conda install -c pytorch pytorch torchvision
conda install -c conda-forge opencv numpy matplotlib pillow tifffile
pip install rasterio
```

## Usage

### Basic Usage with Example Script

The easiest way to use this tool is with the provided example script:

```bash
python example_usage.py --dataset ~/Dropbox/DEV/Data/RedEdge --field 000 --format tif
```

### Arguments for example_usage.py

- `--dataset`: Path to the RedEdge dataset root directory (default: ~/Dropbox/DEV/Data/RedEdge)
- `--field`: Field ID to analyze (000, 001, 002, 003, or 004) (default: 000)
- `--format`: Image format to use (tif or png) (default: tif)
- `--threshold`: NDVI threshold for vegetation detection (default: 0.2)
- `--output-dir`: Directory to save results (default: results)

### Direct Usage

You can also use the main script directly:

```bash
# Using TIF format (reflectance-tif directory)
python crop_counter_health.py --image ~/Dropbox/DEV/Data/RedEdge/000/reflectance-tif --format tif --red-idx 3 --nir-idx 2

# Using PNG format (composite-png directory)
python crop_counter_health.py --image ~/Dropbox/DEV/Data/RedEdge/000/composite-png --format png --red-idx 0 --nir-idx 0
```

### Arguments for crop_counter_health.py

- `--image`: Path to the multispectral image or directory (required)
- `--format`: Image format to use (tif or png) (default: tif)
- `--threshold`: NDVI threshold for vegetation detection (default: 0.2)
- `--red-idx`: Index of the red band in the image (default: 0)
- `--nir-idx`: Index of the NIR band in the image (default: 3)
- `--output`: Output file path for the visualization (default: crop_analysis_results.png)
- `--min-area`: Minimum area (in pixels) for a region to be considered a plant (default: 50)

## Output

The script generates:

1. A visualization image showing:
   - Original RGB image
   - NDVI map
   - Vegetation mask
   - Detected plants with ID numbers and health status

2. Console output with:
   - Total plant count
   - Health statistics summary
   - Detailed information for each plant

## Health Classification

Plants are classified into health categories based on their mean NDVI value:

- **Poor**: NDVI < 0.2
- **Fair**: 0.2 ≤ NDVI < 0.4
- **Good**: 0.4 ≤ NDVI < 0.6
- **Excellent**: NDVI ≥ 0.6

## Tips for Best Results

1. **Use TIF format** for more accurate results, as these contain the full reflectance data
2. **Adjust the NDVI threshold** based on your specific field conditions
3. **Increase the min-area parameter** if you're getting too many small detections
4. **Process one field at a time** for better performance

## Acknowledgments

This tool adapts code from the RoWeeder project, which provides unsupervised weed mapping through crop-row detection.
