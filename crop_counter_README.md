# Crop Counter and Health Assessment Tool

This tool analyzes multispectral drone imagery to count individual crops and assess their health using NDVI (Normalized Difference Vegetation Index).

## Features

- **Crop Counting**: Identifies and counts individual plants using connected component analysis
- **Health Assessment**: Calculates NDVI for each plant and categorizes health status
- **Visualization**: Generates visual output with plant identification and health status

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL (Pillow)

You can install the required packages using:

```bash
pip install torch torchvision opencv-python numpy matplotlib pillow
```

## Usage

```bash
python crop_counter_health.py --image <path_to_image> [options]
```

### Arguments

- `--image`: Path to the multispectral image or directory containing band images (required)
- `--threshold`: NDVI threshold for vegetation detection (default: 0.2)
- `--red-idx`: Index of the red band in the image (default: 0)
- `--nir-idx`: Index of the NIR band in the image (default: 3)

### Input Image Formats

The tool supports two types of input:

1. **Single multispectral file** (e.g., TIFF with multiple bands)
2. **Directory structure** with separate files for each band (R, G, B, NIR)

### Example

```bash
# Using a TIFF file with multiple bands
python crop_counter_health.py --image sample_data/drone_image.tif --red-idx 0 --nir-idx 3

# Using a directory with separate band files
python crop_counter_health.py --image sample_data/field_001/ --red-idx 0 --nir-idx 3
```

## Output

The script generates:

1. A visualization image (`crop_analysis_results.png`) showing:
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

## Adapting for Your Data

You may need to adjust the following parameters based on your specific imagery:

- NDVI threshold: Adjust based on your vegetation density and sensor characteristics
- Red and NIR band indices: These depend on how your multispectral data is organized
- Health classification thresholds: May need calibration for your specific crop type

## Acknowledgments

This tool adapts code from the RoWeeder project, which provides unsupervised weed mapping through crop-row detection.
