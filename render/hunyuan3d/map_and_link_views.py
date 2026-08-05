#!/usr/bin/env python3
"""
Map views from 150-view dataset to 24-view format and create symlinks/copies.
This allows reusing 150-view renders for 24-view training without re-rendering.

Batch processing mode for TRELLIS dataset:
  --dataset-dir: Dataset base directory (e.g., /path/to/Toys4k)
                 If 'renders' is not in the path, it will be automatically appended
                 Output directory is automatically set to {dataset-dir}/renders_hunyuan3d
"""
import json
import numpy as np
import os
import sys
import shutil
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def load_transforms(json_path):
    """Load transforms.json file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def extract_camera_angles(view):
    """Extract azimuth and elevation from view data."""
    if 'azimuth' in view and 'elevation' in view:
        return view['azimuth'], view['elevation']
    elif 'transform_matrix' in view:
        matrix = np.array(view['transform_matrix'])
        cam_pos = matrix[:3, 3]
        cam_dis = np.linalg.norm(cam_pos)
        if cam_dis < 1e-6:
            return 0.0, 0.0
        direction = cam_pos / cam_dis
        elevation = np.arcsin(np.clip(direction[2], -1.0, 1.0))
        azimuth = np.arctan2(direction[1], direction[0])
        return azimuth, elevation
    elif 'hangle' in view and 'vangle' in view:
        # TRELLIS format: hangle (azimuth/yaw), vangle (elevation/pitch)
        return view['hangle'], view['vangle']
    else:
        raise ValueError(f"View missing camera angle information: {list(view.keys())}")

def calculate_angular_distance(az1, el1, az2, el2):
    """Calculate angular distance between two camera positions on sphere"""
    x1 = np.cos(el1) * np.cos(az1)
    y1 = np.cos(el1) * np.sin(az1)
    z1 = np.sin(el1)
    
    x2 = np.cos(el2) * np.cos(az2)
    y2 = np.cos(el2) * np.sin(az2)
    z2 = np.sin(el2)
    
    dot = x1*x2 + y1*y2 + z1*z2
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def generate_standard_24view_reference():
    """
    Generate standard 24-view reference using Hammersley sequence (deterministic).
    This matches the view distribution used in TRELLIS/Hunyuan3D training.
    Uses the same logic as render.py trellis_cond_camera_sequence.
    """
    # Hammersley sequence implementation (from render.py)
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val
    
    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]
    
    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)
    
    def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
        u, v = hammersley_sequence(2, n, num_samples)
        u += offset[0] / num_samples
        v += offset[1]
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
        theta = np.arccos(1 - 2 * u) - np.pi / 2
        phi = v * 2 * np.pi
        return [phi, theta]
    
    # Generate 24 views using Hammersley sequence with fixed offset for determinism
    offset = (0.0, 0.0)  # Fixed offset for reproducible results
    views = []
    for i in range(24):
        phi, theta = sphere_hammersley_sequence(i, 24, offset)
        # phi is azimuth (yaw), theta is elevation (pitch)
        views.append({
            'azimuth': phi,
            'elevation': theta,
            'file_path': f'{i:03d}.png',
            # Also include hangle/vangle for compatibility
            'hangle': phi,
            'vangle': theta
        })
    
    return {'frames': views}

def find_best_matches(views150, views24, logger=None):
    """Find best matching views from 150-view set for each 24-view"""
    mappings = []
    
    for i24, view24 in enumerate(views24):
        az24, el24 = extract_camera_angles(view24)
        
        best_match = None
        best_distance = float('inf')
        
        for i150, view150 in enumerate(views150):
            az150, el150 = extract_camera_angles(view150)
            dist = calculate_angular_distance(az24, el24, az150, el150)
            
            if dist < best_distance:
                best_distance = dist
                best_match = i150
        
        mappings.append({
            'view24_idx': i24,
            'view150_idx': best_match,
            'angular_distance': best_distance
        })
        if logger:
            logger.debug(f"  24-view[{i24:03d}] <-> 150-view[{best_match:03d}] "
                  f"(dist: {np.degrees(best_distance):.2f}°)")
    
    return mappings

def map_single_sha256(sha256_dir, output_dir, reference_24view, use_symlink=True, logger=None):
    """
    Map a single sha256 folder from 150-view to 24-view format.
    
    Args:
        sha256_dir: Directory containing 150-view renders (e.g., .../renders/{sha256}/)
        output_dir: Output directory (e.g., .../renders_hunyuan3d/{sha256}/render_cond/)
        reference_24view: Reference 24-view transforms data (dict with 'frames')
        use_symlink: If True, create symlinks; if False, copy files
        logger: Logger instance for detailed logging
    """
    sha256_name = os.path.basename(sha256_dir)
    transforms150_path = os.path.join(sha256_dir, 'transforms.json')
    if not os.path.exists(transforms150_path):
        if logger:
            logger.warning(f"SKIP {sha256_name}: transforms.json not found")
        return False, None
    
    if logger:
        logger.info(f"Processing {sha256_name}")
        logger.debug(f"  Loading 150-view transforms from {transforms150_path}")
    data150 = load_transforms(transforms150_path)
    views150 = data150['frames']
    
    if len(views150) < 24:
        if logger:
            logger.warning(f"SKIP {sha256_name}: Only {len(views150)} views found (need at least 24)")
        return False, None
    
    if logger:
        logger.debug(f"  Found {len(views150)} views")
        logger.debug(f"  Finding best matches...")
    
    mappings = find_best_matches(views150, reference_24view['frames'], logger=logger)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create mapped files
    if logger:
        logger.debug(f"  Creating {'symlinks' if use_symlink else 'copies'}...")
    for mapping in mappings:
        src_idx = mapping['view150_idx']
        dst_idx = mapping['view24_idx']
        
        # Find source file (could be .png, .jpg, etc.)
        src_patterns = [
            f"{src_idx:03d}.png",
            f"{src_idx:03d}.jpg",
            f"{src_idx:03d}.jpeg"
        ]
        
        src_file = None
        for pattern in src_patterns:
            candidate = os.path.join(sha256_dir, pattern)
            if os.path.exists(candidate):
                src_file = candidate
                break
        
        if src_file is None:
            if logger:
                logger.warning(f"    WARNING: Source file not found for view {src_idx:03d}")
            continue
        
        dst_file = os.path.join(output_dir, f"{dst_idx:03d}.png")
        
        if use_symlink:
            if os.path.exists(dst_file):
                if os.path.islink(dst_file):
                    os.unlink(dst_file)
                else:
                    os.remove(dst_file)
            src_abs = os.path.abspath(src_file)
            os.symlink(src_abs, dst_file)
            if logger:
                logger.debug(f"    Created symlink: {dst_idx:03d}.png -> {src_idx:03d}.png")
        else:
            shutil.copy2(src_file, dst_file)
            if logger:
                logger.debug(f"    Copied: {dst_idx:03d}.png <- {src_idx:03d}.png")
    
    # Create symlink/copy for mesh.ply
    mesh_src = os.path.join(sha256_dir, 'mesh.ply')
    mesh_dst = os.path.join(output_dir, 'mesh.ply')
    
    if os.path.exists(mesh_src):
        if use_symlink:
            if os.path.exists(mesh_dst):
                if os.path.islink(mesh_dst):
                    os.unlink(mesh_dst)
                else:
                    os.remove(mesh_dst)
            mesh_src_abs = os.path.abspath(mesh_src)
            os.symlink(mesh_src_abs, mesh_dst)
            if logger:
                logger.debug(f"    Created symlink: mesh.ply")
        else:
            shutil.copy2(mesh_src, mesh_dst)
            if logger:
                logger.debug(f"    Copied: mesh.ply")
    elif logger:
        logger.warning(f"    WARNING: mesh.ply not found in {sha256_dir}")
    
    # Create new transforms.json for 24-view
    if logger:
        logger.debug(f"  Creating transforms.json...")
    new_transforms = {
        "aabb": data150.get("aabb", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
        "scale": data150.get("scale", 1.0),
        "offset": data150.get("offset", [0.0, 0.0, 0.0]),
        "frames": []
    }
    
    # Sort mappings by view24_idx to ensure correct order
    sorted_mappings = sorted(mappings, key=lambda x: x['view24_idx'])
    
    for mapping in sorted_mappings:
        view150 = views150[mapping['view150_idx']]
        new_frame = view150.copy()
        # Update file_path to match the new 24-view naming
        new_frame['file_path'] = f"{mapping['view24_idx']:03d}.png"
        new_transforms['frames'].append(new_frame)
    
    transforms_output = os.path.join(output_dir, 'transforms.json')
    with open(transforms_output, 'w') as f:
        json.dump(new_transforms, f, indent=4)
    
    if logger:
        avg_dist = np.degrees(np.mean([m['angular_distance'] for m in mappings]))
        max_dist = np.degrees(np.max([m['angular_distance'] for m in mappings]))
        logger.info(f"  Done! (avg distance: {avg_dist:.2f}°, max: {max_dist:.2f}°)")
    
    return True, {
        'avg_distance': np.degrees(np.mean([m['angular_distance'] for m in mappings])),
        'max_distance': np.degrees(np.max([m['angular_distance'] for m in mappings]))
    }

def setup_logging(log_file=None):
    """Setup logging to file and console"""
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"map_and_link_views_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('map_and_link_views')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers
    
    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (only warnings and errors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_file

def process_batch(dataset_dir, hunyuan_dir=None, reference_24view_path=None, use_symlink=True):
    """
    Process all sha256 folders in dataset_dir and create symlinks in hunyuan_dir.
    
    Args:
        dataset_dir: Dataset base directory (e.g., /path/to/Toys4k)
                    If 'renders' is not in the path, it will be automatically appended
        hunyuan_dir: Output directory (creates {sha256}/render_cond/ structure)
                    If None, automatically generated from dataset_dir by appending '_hunyuan3d'
        reference_24view_path: Optional path to reference 24-view transforms.json
                              If None, generates standard 24-view
        use_symlink: If True, create symlinks; if False, copy files
    """
    # Normalize dataset_dir: remove trailing slash
    dataset_dir_clean = dataset_dir.rstrip('/')
    
    # Auto-append 'renders' if not present in the path
    if 'renders' not in dataset_dir_clean:
        trellis_dir = os.path.join(dataset_dir_clean, 'renders')
    else:
        trellis_dir = dataset_dir_clean
    
    # Auto-generate hunyuan_dir if not provided
    if hunyuan_dir is None:
        # Append '_hunyuan3d' to the directory name
        hunyuan_dir = trellis_dir + '_hunyuan3d'
    
    # Setup logging
    log_file = os.path.join(hunyuan_dir, 'map_and_link_views.log')
    os.makedirs(hunyuan_dir, exist_ok=True)
    logger, actual_log_file = setup_logging(log_file)
    
    logger.info(f"=== Batch Processing TRELLIS Dataset ===")
    logger.info(f"Dataset directory: {dataset_dir_clean}")
    logger.info(f"Source (renders): {trellis_dir}")
    logger.info(f"Output: {hunyuan_dir} (auto-generated)")
    logger.info(f"Log file: {actual_log_file}")
    
    print(f"\nProcessing TRELLIS dataset...")
    print(f"  Dataset directory: {dataset_dir_clean}")
    print(f"  Source (renders): {trellis_dir}")
    print(f"  Output: {hunyuan_dir} (auto-generated)")
    print(f"  Log file: {actual_log_file}")
    
    # Load or generate 24-view reference
    if reference_24view_path and os.path.exists(reference_24view_path):
        logger.info(f"Loading reference 24-view from {reference_24view_path}")
        reference_24view = load_transforms(reference_24view_path)
        logger.info(f"Found {len(reference_24view['frames'])} reference views")
    else:
        logger.info("Generating standard 24-view reference")
        reference_24view = generate_standard_24view_reference()
        logger.info("Generated 24 standard views")
    
    # Scan for sha256 folders
    logger.info(f"Scanning for sha256 folders in {trellis_dir}")
    sha256_folders = []
    if not os.path.exists(trellis_dir):
        logger.error(f"Directory does not exist: {trellis_dir}")
        print(f"ERROR: Directory does not exist: {trellis_dir}")
        return
    
    for item in os.listdir(trellis_dir):
        item_path = os.path.join(trellis_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a sha256 hash (64 hex characters)
            if re.match(r'^[0-9a-f]{64}$', item, re.IGNORECASE):
                transforms_path = os.path.join(item_path, 'transforms.json')
                if os.path.exists(transforms_path):
                    sha256_folders.append((item, item_path))
    
    logger.info(f"Found {len(sha256_folders)} sha256 folders with transforms.json")
    print(f"  Found {len(sha256_folders)} sha256 folders")
    
    if len(sha256_folders) == 0:
        logger.warning("No valid sha256 folders found!")
        print("  No valid sha256 folders found!")
        return
    
    # Process each sha256 folder with progress bar
    logger.info(f"Processing {len(sha256_folders)} folders")
    print(f"\nProcessing {len(sha256_folders)} folders...")
    
    success_count = 0
    skip_count = 0
    errors = []
    
    # Use tqdm if available, otherwise simple progress
    if tqdm:
        pbar = tqdm(total=len(sha256_folders), desc="Processing", unit="folder")
    
    for idx, (sha256, sha256_dir) in enumerate(sha256_folders):
        output_dir = os.path.join(hunyuan_dir, sha256, 'render_cond')
        
        try:
            success, stats = map_single_sha256(
                sha256_dir, 
                output_dir, 
                reference_24view, 
                use_symlink=use_symlink,
                logger=logger
            )
            if success:
                success_count += 1
                if tqdm:
                    pbar.set_postfix({
                        'success': success_count,
                        'skip': skip_count,
                        'current': sha256[:8]
                    })
            else:
                skip_count += 1
        except Exception as e:
            error_msg = f"ERROR processing {sha256}: {e}"
            logger.error(error_msg, exc_info=True)
            errors.append((sha256, str(e)))
            skip_count += 1
        
        if tqdm:
            pbar.update(1)
        else:
            # Simple progress without tqdm
            if (idx + 1) % 10 == 0 or idx == len(sha256_folders) - 1:
                print(f"  Progress: {idx + 1}/{len(sha256_folders)} (success: {success_count}, skip: {skip_count})")
    
    if tqdm:
        pbar.close()
    
    # Summary
    logger.info(f"=== Summary ===")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Skipped/Failed: {skip_count}")
    logger.info(f"Total: {len(sha256_folders)}")
    logger.info(f"Output structure: {hunyuan_dir}/{{sha256}}/render_cond/")
    logger.info(f"Files are {'symlinked' if use_symlink else 'copied'} from {trellis_dir}")
    
    if errors:
        logger.warning(f"Errors occurred in {len(errors)} folders:")
        for sha256, error in errors:
            logger.warning(f"  {sha256}: {error}")
    
    print(f"\n=== Summary ===")
    print(f"  Successfully processed: {success_count}")
    print(f"  Skipped/Failed: {skip_count}")
    print(f"  Total: {len(sha256_folders)}")
    print(f"  Detailed log: {actual_log_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Map 150-view TRELLIS renders to 24-view format for Hunyuan3D training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch process all sha256 folders in TRELLIS directory
  # If 'renders' is not in the path, it will be automatically appended
  python map_and_link_views.py \\
    --dataset-dir /mnt/sdc_870evo_8TB/Toys4k
  # Uses: /mnt/sdc_870evo_8TB/Toys4k/renders
  # Output: /mnt/sdc_870evo_8TB/Toys4k/renders_hunyuan3d

  # Use custom 24-view reference file
  python map_and_link_views.py \\
    --dataset-dir /mnt/sdc_870evo_8TB/Toys4k \\
    --reference-24view /path/to/reference/transforms.json

  # Copy files instead of creating symlinks
  python map_and_link_views.py \\
    --dataset-dir /mnt/sdc_870evo_8TB/Toys4k \\
    --copy

Output Structure:
  {dataset-dir}/renders_hunyuan3d/{sha256}/render_cond/
    ├── 000.png (symlink)
    ├── ...
    ├── 023.png (symlink)
    ├── mesh.ply (symlink)
    └── transforms.json (new file)
        """
    )
    
    parser.add_argument(
        '--dataset-dir',
        type=str,
        required=True,
        help='Dataset base directory (e.g., /path/to/Toys4k). If "renders" is not in the path, it will be automatically appended. Output directory will be automatically set to {dataset-dir}/renders_hunyuan3d'
    )
    parser.add_argument(
        '--reference-24view',
        type=str,
        default=None,
        help='Optional path to reference 24-view transforms.json (if not provided, generates standard 24-view)'
    )
    parser.add_argument(
        '--copy',
        action='store_true',
        help='Copy files instead of creating symlinks (default: symlinks)'
    )
    
    args = parser.parse_args()
    
    process_batch(
        args.dataset_dir,
        hunyuan_dir=None,  # Auto-generate from dataset_dir
        reference_24view_path=args.reference_24view,
        use_symlink=not args.copy
    )
