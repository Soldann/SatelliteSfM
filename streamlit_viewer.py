#!/usr/bin/env python3
"""
Streamlit application for visualizing satellite images across different time points
and identifying scenes with the most temporal/construction changes.
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
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_IMAGES_DIR = '/project/jayinnn/SatelliteSfM/data/DFC2019_JAX_preprocessed/images'
DEFAULT_METAS_DIR = '/project/jayinnn/SatelliteSfM/data/DFC2019_JAX_preprocessed/metas'
DEFAULT_THUMBNAILS_DIR = '/project/jayinnn/SatelliteSfM/thumbnails'
DEFAULT_ROTATION_FILE = '/project/jayinnn/SatelliteSfM/rotation_offsets.json'


# ============================================================================
# Utility Functions
# ============================================================================

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


def format_date(dt: datetime) -> str:
    """Format datetime for display."""
    if dt == datetime.min:
        return "Unknown"
    return dt.strftime("%Y-%m-%d %H:%M")


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


def compute_edge_difference(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Compute edge-based difference between two images.
    Returns (difference_score, difference_heatmap).
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Apply Canny edge detection
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)
    
    # Compute difference
    diff = cv2.absdiff(edges1, edges2)
    
    # Dilate to make edges more visible
    kernel = np.ones((3, 3), np.uint8)
    diff = cv2.dilate(diff, kernel, iterations=1)
    
    # Calculate score (normalized)
    score = np.sum(diff > 0) / diff.size
    
    return score, diff


def compute_ssim_difference(img1: np.ndarray, img2: np.ndarray) -> tuple:
    """
    Compute SSIM-based difference between two images.
    Returns (difference_score, difference_heatmap).
    """
    # Convert to grayscale
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # Compute SSIM with difference image
    score, diff_img = ssim(gray1, gray2, full=True)
    
    # Convert to heatmap (invert so differences are bright)
    diff_img = ((1 - diff_img) * 255).astype(np.uint8)
    
    return 1 - score, diff_img


def create_difference_overlay(img: np.ndarray, diff_heatmap: np.ndarray, 
                               alpha: float = 0.5) -> np.ndarray:
    """Create overlay of original image with difference heatmap."""
    # Ensure same size
    if diff_heatmap.shape[:2] != img.shape[:2]:
        diff_heatmap = cv2.resize(diff_heatmap, (img.shape[1], img.shape[0]))
    
    # Create colored heatmap
    heatmap_colored = cv2.applyColorMap(diff_heatmap, cv2.COLORMAP_JET)
    
    # Convert img to BGR if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Create overlay
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


# ============================================================================
# Data Loading
# ============================================================================

@st.cache_data
def load_scene_data(images_dir: str, metas_dir: str, rotation_file: str):
    """Load all scene data and organize by scene ID."""
    # Load rotation offsets if available
    rotation_offsets = {}
    if os.path.exists(rotation_file):
        with open(rotation_file, 'r') as f:
            data = json.load(f)
            rotation_offsets = data.get('rotation_offsets', {})
    
    # Scan all images
    image_files = glob.glob(os.path.join(images_dir, 'JAX_*_*_RGB.png'))
    scenes = defaultdict(list)
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        scene_id, img_id = parse_filename(filename)
        if scene_id:
            # Load metadata
            meta_path = os.path.join(metas_dir, filename.replace('.png', '.json'))
            meta = load_metadata(meta_path)
            capture_date = get_capture_date(meta)
            
            # Get rotation
            key = f"{scene_id}_{img_id}"
            rotation = rotation_offsets.get(key, 0)
            
            scenes[scene_id].append({
                'path': img_path,
                'filename': filename,
                'img_id': img_id,
                'capture_date': capture_date,
                'rotation': rotation,
                'meta': meta
            })
    
    # Sort by capture date
    for scene_id in scenes:
        scenes[scene_id].sort(key=lambda x: x['capture_date'])
    
    return dict(scenes), rotation_offsets


@st.cache_data
def compute_scene_change_scores(scenes: dict, _images_dir: str, sample_size: int = 512):
    """Compute temporal change scores for each scene."""
    change_scores = {}
    
    for scene_id, images in scenes.items():
        if len(images) < 2:
            change_scores[scene_id] = 0.0
            continue
        
        total_score = 0.0
        num_pairs = 0
        
        # Compare consecutive pairs
        for i in range(len(images) - 1):
            img1_path = images[i]['path']
            img2_path = images[i + 1]['path']
            
            # Load images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                continue
            
            # Resize for faster computation
            img1 = cv2.resize(img1, (sample_size, sample_size))
            img2 = cv2.resize(img2, (sample_size, sample_size))
            
            # Apply rotations
            img1 = rotate_image(img1, images[i]['rotation'])
            img2 = rotate_image(img2, images[i + 1]['rotation'])
            
            # Compute difference
            score, _ = compute_edge_difference(img1, img2)
            total_score += score
            num_pairs += 1
        
        change_scores[scene_id] = total_score / num_pairs if num_pairs > 0 else 0.0
    
    return change_scores


def load_image(img_info: dict, rotation: int = None, max_size: int = None) -> np.ndarray:
    """Load and optionally rotate/resize an image."""
    img = cv2.imread(img_info['path'])
    if img is None:
        return None
    
    # Convert BGR to RGB for display
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply rotation
    if rotation is None:
        rotation = img_info.get('rotation', 0)
    img = rotate_image(img, rotation)
    
    # Resize if needed
    if max_size and max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size)
    
    return img


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="🛰️ Satellite Temporal Viewer",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    .scene-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .change-score-high {
        background-color: #ff4444;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .change-score-medium {
        background-color: #ffaa00;
        color: black;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .change-score-low {
        background-color: #44aa44;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">🛰️ Satellite Temporal Change Viewer</p>', 
                unsafe_allow_html=True)
    
    # Sidebar - Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        images_dir = st.text_input("Images Directory", DEFAULT_IMAGES_DIR)
        metas_dir = st.text_input("Metadata Directory", DEFAULT_METAS_DIR)
        rotation_file = st.text_input("Rotation File", DEFAULT_ROTATION_FILE)
        
        if not os.path.exists(images_dir):
            st.error(f"Images directory not found: {images_dir}")
            st.stop()
        
        # Load data
        with st.spinner("Loading scene data..."):
            scenes, rotation_offsets = load_scene_data(images_dir, metas_dir, rotation_file)
        
        st.success(f"Loaded {len(scenes)} scenes")
        
        # Compute change scores
        compute_scores = st.checkbox("Compute Change Scores", value=False,
                                     help="Warning: This may take a while for first run")
        
        change_scores = {}
        if compute_scores:
            with st.spinner("Computing change scores..."):
                change_scores = compute_scene_change_scores(scenes, images_dir)
            st.success("Change scores computed!")
        
        st.divider()
        
        # Scene selection
        st.header("📍 Scene Selection")
        
        sort_option = st.selectbox(
            "Sort scenes by:",
            ["Scene ID", "Number of Images", "Change Score (High to Low)"]
        )
        
        # Sort scenes
        scene_list = list(scenes.keys())
        if sort_option == "Scene ID":
            scene_list.sort(key=lambda x: int(x))
        elif sort_option == "Number of Images":
            scene_list.sort(key=lambda x: len(scenes[x]), reverse=True)
        elif sort_option == "Change Score (High to Low)" and change_scores:
            scene_list.sort(key=lambda x: change_scores.get(x, 0), reverse=True)
        
        # Display scene selector with info
        scene_options = []
        for scene_id in scene_list:
            num_images = len(scenes[scene_id])
            score_str = ""
            if scene_id in change_scores:
                score = change_scores[scene_id]
                score_str = f" | Score: {score:.3f}"
            scene_options.append(f"JAX_{scene_id} ({num_images} imgs{score_str})")
        
        selected_option = st.selectbox("Select Scene:", scene_options)
        selected_scene = scene_list[scene_options.index(selected_option)]
        
        st.divider()
        
        # Display options
        st.header("🎨 Display Options")
        
        display_mode = st.radio(
            "View Mode:",
            ["Side-by-Side", "Timeline Gallery", "Difference Analysis"]
        )
        
        show_rotation_info = st.checkbox("Show rotation info", value=True)
        max_display_size = st.slider("Max image size (px)", 256, 1024, 512)
    
    # Main content area
    scene_images = scenes[selected_scene]
    
    # Scene info header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📸 Scene: JAX_{selected_scene}")
    with col2:
        st.metric("Total Images", len(scene_images))
    with col3:
        if selected_scene in change_scores:
            score = change_scores[selected_scene]
            st.metric("Change Score", f"{score:.4f}")
    
    # Date range
    if scene_images:
        dates = [img['capture_date'] for img in scene_images if img['capture_date'] != datetime.min]
        if dates:
            st.caption(f"📅 Date range: {format_date(min(dates))} → {format_date(max(dates))}")
    
    st.divider()
    
    # ========================================
    # Display Mode: Side-by-Side
    # ========================================
    if display_mode == "Side-by-Side":
        if len(scene_images) < 2:
            st.warning("Need at least 2 images for comparison")
        else:
            # Select two time points
            col1, col2 = st.columns(2)
            
            with col1:
                idx1 = st.selectbox(
                    "Time Point 1 (Earlier)",
                    range(len(scene_images)),
                    format_func=lambda i: f"{i+1}. {format_date(scene_images[i]['capture_date'])} (ID: {scene_images[i]['img_id']})"
                )
            
            with col2:
                idx2 = st.selectbox(
                    "Time Point 2 (Later)",
                    range(len(scene_images)),
                    index=min(len(scene_images)-1, idx1+1),
                    format_func=lambda i: f"{i+1}. {format_date(scene_images[i]['capture_date'])} (ID: {scene_images[i]['img_id']})"
                )
            
            # Load images
            img1 = load_image(scene_images[idx1], max_size=max_display_size)
            img2 = load_image(scene_images[idx2], max_size=max_display_size)
            
            if img1 is not None and img2 is not None:
                # Display side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(img1, caption=f"Time Point 1: {format_date(scene_images[idx1]['capture_date'])}")
                    if show_rotation_info:
                        st.caption(f"Rotation: {scene_images[idx1]['rotation']}°")
                
                with col2:
                    st.image(img2, caption=f"Time Point 2: {format_date(scene_images[idx2]['capture_date'])}")
                    if show_rotation_info:
                        st.caption(f"Rotation: {scene_images[idx2]['rotation']}°")
                
                # Show difference
                st.subheader("📊 Difference Analysis")
                
                diff_method = st.radio("Difference Method:", 
                                       ["Edge-based", "SSIM"], 
                                       horizontal=True)
                
                # Compute difference
                img1_cv = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                img2_cv = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                
                if diff_method == "Edge-based":
                    score, diff_heatmap = compute_edge_difference(img1_cv, img2_cv)
                else:
                    score, diff_heatmap = compute_ssim_difference(img1_cv, img2_cv)
                
                # Create overlay
                overlay_alpha = st.slider("Overlay intensity", 0.0, 1.0, 0.4)
                overlay = create_difference_overlay(img1_cv, diff_heatmap, overlay_alpha)
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(overlay, caption=f"Difference Overlay (Score: {score:.4f})")
    
    # ========================================
    # Display Mode: Timeline Gallery
    # ========================================
    elif display_mode == "Timeline Gallery":
        st.subheader("📅 Timeline View")
        
        # Slider to navigate through time
        if len(scene_images) > 1:
            # Create options with formatted labels
            time_options = list(range(len(scene_images)))
            time_idx = st.select_slider(
                "Navigate Timeline",
                options=time_options,
                value=0,
                format_func=lambda i: format_date(scene_images[i]['capture_date'])
            )
        else:
            time_idx = 0
        
        # Display current image
        img = load_image(scene_images[time_idx], max_size=max_display_size)
        if img is not None:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img)
                st.caption(f"📅 {format_date(scene_images[time_idx]['capture_date'])} | "
                          f"ID: {scene_images[time_idx]['img_id']} | "
                          f"Rotation: {scene_images[time_idx]['rotation']}°")
        
        # Thumbnail gallery
        st.subheader("🖼️ All Time Points")
        
        cols_per_row = st.slider("Columns", 3, 8, 5)
        thumb_size = 150
        
        rows = (len(scene_images) + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < len(scene_images):
                    with cols[col_idx]:
                        thumb = load_image(scene_images[img_idx], max_size=thumb_size)
                        if thumb is not None:
                            st.image(thumb, use_container_width=True)
                            st.caption(f"{scene_images[img_idx]['img_id']}")
    
    # ========================================
    # Display Mode: Difference Analysis
    # ========================================
    elif display_mode == "Difference Analysis":
        st.subheader("📈 Temporal Change Analysis")
        
        if len(scene_images) < 2:
            st.warning("Need at least 2 images for analysis")
        else:
            # Compute pairwise differences
            st.write("Computing pairwise differences between consecutive images...")
            
            differences = []
            for i in range(len(scene_images) - 1):
                img1 = load_image(scene_images[i], max_size=512)
                img2 = load_image(scene_images[i + 1], max_size=512)
                
                if img1 is not None and img2 is not None:
                    img1_cv = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                    img2_cv = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    score, _ = compute_edge_difference(img1_cv, img2_cv)
                    
                    differences.append({
                        'from_idx': i,
                        'to_idx': i + 1,
                        'from_date': scene_images[i]['capture_date'],
                        'to_date': scene_images[i + 1]['capture_date'],
                        'score': score
                    })
            
            if differences:
                # Find max change period
                max_diff = max(differences, key=lambda x: x['score'])
                
                st.success(f"🎯 **Maximum change detected between:** "
                          f"{format_date(max_diff['from_date'])} → {format_date(max_diff['to_date'])} "
                          f"(Score: {max_diff['score']:.4f})")
                
                # Bar chart of changes
                import pandas as pd
                df = pd.DataFrame([{
                    'Period': f"{d['from_idx']+1}→{d['to_idx']+1}",
                    'Change Score': d['score']
                } for d in differences])
                
                st.bar_chart(df.set_index('Period'))
                
                # Show maximum change images
                st.subheader("🔍 Maximum Change Period")
                
                col1, col2 = st.columns(2)
                
                img1 = load_image(scene_images[max_diff['from_idx']], max_size=max_display_size)
                img2 = load_image(scene_images[max_diff['to_idx']], max_size=max_display_size)
                
                with col1:
                    if img1 is not None:
                        st.image(img1, caption=f"Before: {format_date(max_diff['from_date'])}")
                
                with col2:
                    if img2 is not None:
                        st.image(img2, caption=f"After: {format_date(max_diff['to_date'])}")


if __name__ == '__main__':
    main()
