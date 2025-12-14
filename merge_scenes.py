#!/usr/bin/env python3
"""
Merge multiple scenes into a single input for SatelliteSfM.

This script combines images and bounding boxes from multiple scenes to create
a unified input directory. This ensures a consistent coordinate system (observer)
when running the SfM pipeline.
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

def find_scene_images(track_rgb_dirs: List[Path], scene_id: str, city: str) -> List[Path]:
    """Find all TIF images matching the scene ID in any of the track directories."""
    pattern = f"{city}_{scene_id}_*_RGB.tif"
    all_images = []
    
    for track_dir in track_rgb_dirs:
        if track_dir.exists():
            images = sorted(track_dir.glob(pattern))
            all_images.extend(images)
            
    if not all_images:
        print(f"Warning: No images found for scene {scene_id} in {track_rgb_dirs}")
        
    return all_images

def find_scene_bbx(preprocessed_dir: Path, scene_id: str, city: str) -> Path:
    """Find a bounding box JSON file matching the scene ID."""
    bbx_dir = preprocessed_dir / "latlonalt_bbx"
    pattern = f"{city}_{scene_id}_*_RGB.json"
    bbx_files = sorted(bbx_dir.glob(pattern))
    
    if not bbx_files:
        print(f"Warning: No bounding box files found for scene {scene_id}")
        return None
    
    return bbx_files[0]

def merge_scenes(
    scenes: List[str],
    output_name: str,
    track_rgb_dirs: List[Path],
    preprocessed_dir: Path,
    output_base_dir: Path,
    city: str = "JAX",
    symlink: bool = False
):
    # Setup output directories
    output_dir = output_base_dir / output_name / "inputs"
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Merging Scenes: {', '.join(scenes)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    all_lat_min = []
    all_lat_max = []
    all_lon_min = []
    all_lon_max = []
    all_alt_min = []
    all_alt_max = []
    
    total_images = 0
    
    for scene_id in scenes:
        print(f"\nProcessing Scene: {scene_id}")
        
        # 1. Handle Images
        images = find_scene_images(track_rgb_dirs, scene_id, city)
        print(f"  Found {len(images)} images")
        
        for img_path in images:
            dest_path = images_dir / img_path.name
            if symlink:
                if dest_path.exists():
                    dest_path.unlink()
                dest_path.symlink_to(img_path.absolute())
            else:
                shutil.copy2(img_path, dest_path)
        
        total_images += len(images)
        
        # 2. Handle Bounding Box
        bbx_path = find_scene_bbx(preprocessed_dir, scene_id, city)
        if bbx_path:
            with open(bbx_path, 'r') as f:
                bbx = json.load(f)
                
            all_lat_min.append(bbx['lat_minmax'][0])
            all_lat_max.append(bbx['lat_minmax'][1])
            all_lon_min.append(bbx['lon_minmax'][0])
            all_lon_max.append(bbx['lon_minmax'][1])
            all_alt_min.append(bbx['alt_minmax'][0])
            all_alt_max.append(bbx['alt_minmax'][1])
            print(f"  Added bounding box from {bbx_path.name}")
        else:
            print(f"  ⚠ Skipping bounding box for {scene_id} (not found)")

    # 3. Create Merged Bounding Box
    if all_lat_min:
        merged_bbx = {
            'lat_minmax': [min(all_lat_min), max(all_lat_max)],
            'lon_minmax': [min(all_lon_min), max(all_lon_max)],
            'alt_minmax': [min(all_alt_min), max(all_alt_max)]
        }
        
        bbx_out_path = output_dir / "latlonalt_bbx.json"
        with open(bbx_out_path, 'w') as f:
            json.dump(merged_bbx, f, indent=2)
            
        print(f"\nCreated merged bounding box: {bbx_out_path}")
        print(f"  Lat: {merged_bbx['lat_minmax']}")
        print(f"  Lon: {merged_bbx['lon_minmax']}")
        print(f"  Alt: {merged_bbx['alt_minmax']}")
    else:
        print("\n⚠ Error: No bounding boxes found to merge!")
        return

    print(f"\n✓ Merge complete!")
    print(f"  Total images: {total_images}")
    print(f"  Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Merge multiple scenes for SatelliteSfM")
    parser.add_argument('--scenes', nargs='+', required=True, help='List of scene IDs (e.g. 004 005)')
    parser.add_argument('--city', type=str, default='JAX', help='City code (JAX or OMA)')
    parser.add_argument('--output_name', type=str, required=True, help='Name of output folder (e.g. JAX_merged)')
    parser.add_argument('--symlink', action='store_true', help='Use symlinks instead of copying')
    
    args = parser.parse_args()
    
    # Default paths
    track_rgb_dirs = [
        Path('data/Track3-RGB-1'),
        Path('data/Track3-RGB-2')
    ]
    preprocessed_dir = Path(f'data/DFC2019_{args.city}_preprocessed')
    output_base_dir = Path('data/DFC2019_processed')
    
    merge_scenes(
        scenes=args.scenes,
        output_name=args.output_name,
        track_rgb_dirs=track_rgb_dirs,
        preprocessed_dir=preprocessed_dir,
        output_base_dir=output_base_dir,
        city=args.city,
        symlink=args.symlink
    )

if __name__ == '__main__':
    main()
