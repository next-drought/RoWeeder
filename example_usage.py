"""
Example script demonstrating how to use the crop_counter_health.py tool
with the RedEdge multispectral imagery dataset.

This script is specifically designed to work with the WeedMap dataset structure:
/path/to/RedEdge/
├── 000, 001, 002, 003, 004 (field folders)
│   ├── composite-png
│   ├── groundtruth
│   └── reflectance-tif
"""

import os
import subprocess
import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description='RedEdge imagery crop counter and health assessment')
    parser.add_argument('--dataset', type=str, default='~/Dropbox/DEV/Data/RedEdge',
                        help='Path to the RedEdge dataset root directory')
    parser.add_argument('--field', type=str, default='000',
                        help='Field ID to analyze (000, 001, 002, 003, or 004)')
    parser.add_argument('--format', type=str, choices=['tif', 'png'], default='tif',
                        help='Image format to use (tif or png)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='NDVI threshold for vegetation detection')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')

    args = parser.parse_args()

    # Expand user directory if needed
    dataset_path = os.path.expanduser(args.dataset)
    field_path = os.path.join(dataset_path, args.field)

    # Check if the dataset exists
    if not os.path.exists(field_path):
        print(f"Error: The specified field path '{field_path}' does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up paths based on format choice
    if args.format == 'tif':
        # For TIF format, we'll create a temporary directory with the bands we need
        temp_dir = os.path.join(args.output_dir, f"temp_{args.field}")
        os.makedirs(temp_dir, exist_ok=True)

        # Find and copy the necessary TIF files
        red_path = os.path.join(field_path, 'reflectance-tif', 'transparent_reflectance_red.tif')
        nir_path = os.path.join(field_path, 'reflectance-tif', 'transparent_reflectance_nir.tif')
        green_path = os.path.join(field_path, 'reflectance-tif', 'transparent_reflectance_green.tif')
        blue_path = os.path.join(field_path, 'reflectance-tif', 'transparent_reflectance_blue.tif')

        # Check if all files exist
        for path in [red_path, nir_path, green_path, blue_path]:
            if not os.path.exists(path):
                print(f"Error: File {path} does not exist.")
                return

        # For TIF files, we'll use the reflectance-tif directory directly
        image_path = os.path.join(field_path, 'reflectance-tif')

        # For TIF files, NIR is at index 2 and Red is at index 3 when loaded as a stack
        red_idx = 3
        nir_idx = 2
    else:
        # For PNG format, we'll use the composite-png directory
        image_path = os.path.join(field_path, 'composite-png')

        # For PNG files, NIR is a separate file, so we'll use indices 0 for both
        # The script will handle loading the correct files
        red_idx = 0
        nir_idx = 0

    # Run the crop counter script
    output_file = os.path.join(args.output_dir, f"field_{args.field}_analysis.png")

    cmd = [
        "python", "crop_counter_health.py",
        "--image", image_path,
        "--threshold", str(args.threshold),
        "--red-idx", str(red_idx),
        "--nir-idx", str(nir_idx),
        "--output", output_file,
        "--format", args.format
    ]

    print("Running crop counter with the following command:")
    print(" ".join(cmd))
    print("\n" + "="*50 + "\n")

    subprocess.run(cmd)

    print("\n" + "="*50)
    print(f"\nAnalysis complete for field {args.field}!")
    print(f"The results have been saved to '{output_file}'")
    print("You can view detailed plant statistics in the console output above.")

if __name__ == "__main__":
    main()
