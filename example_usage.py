"""
Example script demonstrating how to use the crop_counter_health.py tool
with a sample multispectral image.

This script assumes you have a sample multispectral image in the format
required by the tool. If you don't have one, you can download sample data
from various sources like:
- NASA AVIRIS data: https://aviris.jpl.nasa.gov/data/free_data.html
- USGS EarthExplorer: https://earthexplorer.usgs.gov/
"""

import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Example usage of crop counter and health assessment tool')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to sample multispectral image or directory')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='NDVI threshold for vegetation detection')
    parser.add_argument('--red-idx', type=int, default=0,
                        help='Index of the red band in the image')
    parser.add_argument('--nir-idx', type=int, default=3,
                        help='Index of the NIR band in the image')
    
    args = parser.parse_args()
    
    # Check if the image exists
    if not os.path.exists(args.image):
        print(f"Error: The specified image path '{args.image}' does not exist.")
        return
    
    # Run the crop counter script
    cmd = [
        "python", "crop_counter_health.py",
        "--image", args.image,
        "--threshold", str(args.threshold),
        "--red-idx", str(args.red_idx),
        "--nir-idx", str(args.nir_idx)
    ]
    
    print("Running crop counter with the following command:")
    print(" ".join(cmd))
    print("\n" + "="*50 + "\n")
    
    subprocess.run(cmd)
    
    print("\n" + "="*50)
    print("\nAnalysis complete!")
    print("The results have been saved to 'crop_analysis_results.png'")
    print("You can view detailed plant statistics in the console output above.")

if __name__ == "__main__":
    main()
