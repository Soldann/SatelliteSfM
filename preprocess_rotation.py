#!/usr/bin/env python3
"""
Preprocessing script to compute rotation alignments for satellite images.
Uses ORB feature matching to find best rotation (0, 90, 180, 270 degrees)
to align images within each scene to a common reference.
"""

import os
import json
import glob
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def parse_filename(filename: str) -> tuple:
    """Extract scene ID and image ID from filename like JAX_004_006_RGB.png"""
    match = re.match(r'JAX_(\d+)_(\d+)_RGB\.png', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def load_metadata(meta_path: str) -> dict:
    """Load metadata JSON file."""
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except:
        return None


def get_capture_date(meta: dict) -> datetime:
    """Extract capture date from metadata."""
    if meta and 'capture_date' in meta:
        d = meta['capture_date']
        return datetime(d[0], d[1], d[2], d[3], d[4], d[5])
    return datetime.min


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image by angle (0, 90, 180, 270 degrees)."""
    if angle == 0:
        return img
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def compute_ncc_score(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation between two images.
    More robust than feature matching for satellite imagery.
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray1 = img1.astype(np.float32)
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray2 = img2.astype(np.float32)
    
    # Normalize
    gray1 = (gray1 - gray1.mean()) / (gray1.std() + 1e-8)
    gray2 = (gray2 - gray2.mean()) / (gray2.std() + 1e-8)
    
    # Compute correlation
    ncc = np.mean(gray1 * gray2)
    return ncc


def compute_edge_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute edge-based correlation between two images.
    More robust to lighting/seasonal changes in satellite imagery.
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Compute edges using Canny
    edges1 = cv2.Canny(gray1, 50, 150).astype(np.float32)
    edges2 = cv2.Canny(gray2, 50, 150).astype(np.float32)
    
    # Dilate edges slightly for better matching
    kernel = np.ones((3, 3), np.uint8)
    edges1 = cv2.dilate(edges1, kernel, iterations=1)
    edges2 = cv2.dilate(edges2, kernel, iterations=1)
    
    # Normalize
    edges1 = edges1 / (edges1.max() + 1e-8)
    edges2 = edges2 / (edges2.max() + 1e-8)
    
    # Compute overlap score
    intersection = np.sum(edges1 * edges2)
    union = np.sum(edges1) + np.sum(edges2) - intersection + 1e-8
    iou = intersection / union
    
    return iou


def compute_histogram_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute histogram correlation - useful for checking overall color distribution match.
    """
    if len(img1.shape) == 3:
        # Compare histograms for each channel
        scores = []
        for i in range(3):
            hist1 = cv2.calcHist([img1], [i], None, [64], [0, 256])
            hist2 = cv2.calcHist([img2], [i], None, [64], [0, 256])
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            scores.append(score)
        return np.mean(scores)
    else:
        hist1 = cv2.calcHist([img1], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([img2], [0], None, [64], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def compute_feature_match_score(img1: np.ndarray, img2: np.ndarray, 
                                 max_features: int = 1000) -> float:
    """
    Compute feature matching score between two images using SIFT (preferred) or ORB.
    Uses ratio test for more robust matching.
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Try SIFT first (more robust), fall back to ORB
    try:
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    except:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING
    
    # Find keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
        return 0.0
    
    # Match descriptors using BFMatcher with ratio test
    bf = cv2.BFMatcher(norm_type)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except:
        return 0.0
    
    if len(matches) == 0:
        return 0.0
    
    # Apply Lowe's ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    # Score based on number of good matches
    score = len(good_matches)
    
    return score


def find_best_rotation(ref_img: np.ndarray, target_img: np.ndarray, 
                       thumbnail_size: int = 512) -> tuple:
    """
    Find the best rotation angle to align target_img with ref_img.
    Uses multiple methods and voting for robust detection.
    Returns (best_angle, best_score, method_votes).
    """
    # Resize for faster computation
    ref_small = cv2.resize(ref_img, (thumbnail_size, thumbnail_size))
    target_small = cv2.resize(target_img, (thumbnail_size, thumbnail_size))
    
    angles = [0, 90, 180, 270]
    
    # Compute scores for each method
    ncc_scores = []
    edge_scores = []
    feature_scores = []
    hist_scores = []
    
    for angle in angles:
        rotated = rotate_image(target_small, angle)
        
        ncc_scores.append(compute_ncc_score(ref_small, rotated))
        edge_scores.append(compute_edge_correlation(ref_small, rotated))
        feature_scores.append(compute_feature_match_score(ref_small, rotated))
        hist_scores.append(compute_histogram_correlation(ref_small, rotated))
    
    # Convert to numpy for easier manipulation
    ncc_scores = np.array(ncc_scores)
    edge_scores = np.array(edge_scores)
    feature_scores = np.array(feature_scores)
    hist_scores = np.array(hist_scores)
    
    # Vote based on each method's best angle
    votes = np.zeros(4)
    
    # NCC vote (weight: 3) - very reliable for same scene
    votes[np.argmax(ncc_scores)] += 3
    
    # Edge correlation vote (weight: 2) - robust to lighting
    votes[np.argmax(edge_scores)] += 2
    
    # Feature matching vote (weight: 2)
    if np.max(feature_scores) > 10:  # Only count if enough matches
        votes[np.argmax(feature_scores)] += 2
    
    # Histogram vote (weight: 1) - least reliable but useful as tiebreaker
    votes[np.argmax(hist_scores)] += 1
    
    # Get the winning angle
    best_idx = np.argmax(votes)
    best_angle = angles[best_idx]
    
    # Compute confidence score
    total_votes = votes.sum()
    confidence = votes[best_idx] / total_votes if total_votes > 0 else 0
    
    # Combined score for logging
    combined_score = (ncc_scores[best_idx] + edge_scores[best_idx] + 
                     (feature_scores[best_idx] / 100) + hist_scores[best_idx]) / 4
    
    return best_angle, combined_score, {
        'votes': votes.tolist(),
        'ncc': ncc_scores.tolist(),
        'edge': edge_scores.tolist(),
        'features': feature_scores.tolist(),
        'hist': hist_scores.tolist(),
        'confidence': confidence
    }


def create_thumbnails(images_dir: str, output_dir: str, 
                      rotation_offsets: dict, thumbnail_size: int = 256):
    """Create pre-rotated thumbnails for faster loading in Streamlit."""
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = glob.glob(os.path.join(images_dir, 'JAX_*_*_RGB.png'))
    
    for img_path in tqdm(image_files, desc="Creating thumbnails"):
        filename = os.path.basename(img_path)
        scene_id, img_id = parse_filename(filename)
        
        if scene_id is None:
            continue
        
        # Load and resize
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        thumb = cv2.resize(img, (thumbnail_size, thumbnail_size))
        
        # Apply rotation if available
        key = f"{scene_id}_{img_id}"
        if key in rotation_offsets:
            thumb = rotate_image(thumb, rotation_offsets[key])
        
        # Save thumbnail
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, thumb)


def main():
    parser = argparse.ArgumentParser(description='Precompute rotation alignments for satellite images')
    parser.add_argument('--images-dir', type=str, 
                        default='/project/jayinnn/SatelliteSfM/data/DFC2019_JAX_preprocessed/images',
                        help='Directory containing satellite images')
    parser.add_argument('--metas-dir', type=str,
                        default='/project/jayinnn/SatelliteSfM/data/DFC2019_JAX_preprocessed/metas',
                        help='Directory containing metadata JSON files')
    parser.add_argument('--output', type=str,
                        default='/project/jayinnn/SatelliteSfM/rotation_offsets.json',
                        help='Output JSON file for rotation offsets')
    parser.add_argument('--thumbnails-dir', type=str,
                        default='/project/jayinnn/SatelliteSfM/thumbnails',
                        help='Directory to save pre-rotated thumbnails')
    parser.add_argument('--thumbnail-size', type=int, default=256,
                        help='Size of thumbnails')
    parser.add_argument('--skip-thumbnails', action='store_true',
                        help='Skip thumbnail generation')
    args = parser.parse_args()
    
    print(f"Scanning images in: {args.images_dir}")
    
    # Group images by scene
    image_files = glob.glob(os.path.join(args.images_dir, 'JAX_*_*_RGB.png'))
    scenes = defaultdict(list)
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        scene_id, img_id = parse_filename(filename)
        if scene_id:
            # Load metadata for capture date
            meta_path = os.path.join(args.metas_dir, filename.replace('.png', '.json'))
            meta = load_metadata(meta_path)
            capture_date = get_capture_date(meta)
            scenes[scene_id].append({
                'path': img_path,
                'filename': filename,
                'img_id': img_id,
                'capture_date': capture_date
            })
    
    print(f"Found {len(scenes)} scenes with {len(image_files)} total images")
    
    # Sort images in each scene by capture date
    for scene_id in scenes:
        scenes[scene_id].sort(key=lambda x: x['capture_date'])
    
    # Compute rotation offsets for each scene
    rotation_offsets = {}
    scene_stats = {}
    
    for scene_id in tqdm(scenes, desc="Processing scenes"):
        scene_images = scenes[scene_id]
        if len(scene_images) < 2:
            # Single image scene - no rotation needed
            for img_info in scene_images:
                key = f"{scene_id}_{img_info['img_id']}"
                rotation_offsets[key] = 0
            continue
        
        # Use first image (earliest) as reference
        ref_info = scene_images[0]
        ref_img = cv2.imread(ref_info['path'])
        
        if ref_img is None:
            continue
        
        # Reference image always has 0 rotation
        ref_key = f"{scene_id}_{ref_info['img_id']}"
        rotation_offsets[ref_key] = 0
        
        scene_rotations = [0]
        
        # Find best rotation for other images
        for img_info in scene_images[1:]:
            target_img = cv2.imread(img_info['path'])
            if target_img is None:
                continue
            
            best_angle, best_score, vote_info = find_best_rotation(ref_img, target_img)
            key = f"{scene_id}_{img_info['img_id']}"
            rotation_offsets[key] = best_angle
            scene_rotations.append(best_angle)
            
            # Log low confidence detections
            if vote_info['confidence'] < 0.6:
                print(f"  Warning: Low confidence ({vote_info['confidence']:.2f}) for {key}, votes: {vote_info['votes']}")
        
        # Track scene statistics
        scene_stats[scene_id] = {
            'num_images': len(scene_images),
            'rotations': scene_rotations,
            'unique_rotations': len(set(scene_rotations))
        }
    
    # Save rotation offsets
    output_data = {
        'rotation_offsets': rotation_offsets,
        'scene_stats': scene_stats,
        'metadata': {
            'images_dir': args.images_dir,
            'metas_dir': args.metas_dir,
            'num_scenes': len(scenes),
            'num_images': len(image_files)
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\nSaved rotation offsets to: {args.output}")
    print(f"Scenes processed: {len(scenes)}")
    
    # Print rotation distribution
    rotation_counts = defaultdict(int)
    for angle in rotation_offsets.values():
        rotation_counts[angle] += 1
    
    print("\nRotation distribution:")
    for angle in sorted(rotation_counts.keys()):
        print(f"  {angle}°: {rotation_counts[angle]} images")
    
    # Generate thumbnails
    if not args.skip_thumbnails:
        print(f"\nGenerating thumbnails in: {args.thumbnails_dir}")
        create_thumbnails(args.images_dir, args.thumbnails_dir, 
                         rotation_offsets, args.thumbnail_size)
        print("Thumbnail generation complete!")
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()
