#!/bin/bash
# Script to render with multiple lighting conditions for Paint model training
# Processes all files in metadata.csv
# Usage: ./render_lighting_conditions.sh <raw_data_base_dir> [blender_path]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <raw_data_base_dir> [blender_path]"
    echo ""
    echo "Arguments:"
    echo "  raw_data_base_dir   : Base directory containing metadata.csv and raw 3D files"
    echo "                        (e.g., /mnt/sdc_870evo_8TB/Toys4k)"
    echo "                        - metadata.csv should be at: <raw_data_base_dir>/metadata.csv"
    echo "                        - output will be at: <raw_data_base_dir>/renders_hunyuan3d"
    echo "  blender_path        : Optional path to Blender executable (default: blender)"
    echo ""
    echo "Example:"
    echo "  $0 /mnt/sdc_870evo_8TB/Toys4k \\"
    echo "     /home/blender/blender-4.1.1-linux-x64/blender"
    exit 1
fi

RAW_DATA_BASE_DIR="$1"
BLENDER_PATH="${2:-blender}"

# Derive paths from raw_data_base_dir
METADATA_CSV="$RAW_DATA_BASE_DIR/metadata.csv"
OUTPUT_BASE_DIR="$RAW_DATA_BASE_DIR/renders_hunyuan3d"

# Check if raw data base directory exists
if [ ! -d "$RAW_DATA_BASE_DIR" ]; then
    echo "Error: Raw data base directory not found: $RAW_DATA_BASE_DIR"
    exit 1
fi

# Check if metadata.csv exists
if [ ! -f "$METADATA_CSV" ]; then
    echo "Error: Metadata CSV file not found: $METADATA_CSV"
    echo "Expected location: $METADATA_CSV"
    exit 1
fi

# Check if Blender exists
if [ ! -f "$BLENDER_PATH" ] && ! command -v "$BLENDER_PATH" &> /dev/null; then
    echo "Error: Blender not found: $BLENDER_PATH"
    exit 1
fi

# Create output base directory
mkdir -p "$OUTPUT_BASE_DIR"

# Create log file with timestamp in raw_data_base_dir
LOG_FILE="$RAW_DATA_BASE_DIR/render_lighting_conditions_$(date +%Y%m%d_%H%M%S).log"
touch "$LOG_FILE"

echo "=== Batch Rendering Configuration ==="
echo "Raw Data Base Dir: $RAW_DATA_BASE_DIR"
echo "Metadata CSV: $METADATA_CSV"
echo "Output Base Dir: $OUTPUT_BASE_DIR"
echo "Blender Path: $BLENDER_PATH"
echo "Log File: $LOG_FILE"
echo ""

# Function to process a single file
process_single_file() {
    local file_identifier="$1"
    local sha256="$2"
    local input_file="$3"
    local output_base="$4"
    
    echo "----------------------------------------"
    echo "Processing: $file_identifier"
    echo "SHA256: $sha256"
    echo "Input: $input_file"
    echo "Output: $output_base"
    
    if [ ! -f "$input_file" ]; then
        echo "✗ SKIP: Input file not found: $input_file"
        return 1
    fi
    
    # Create output directories
    mkdir -p "$output_base/render_cond"
    mkdir -p "$output_base/render_tex"
    mkdir -p "$output_base/geo_data"
    
    # Render render_tex folder (PBR materials)
    echo "  → Rendering PBR materials to render_tex..."
    $BLENDER_PATH -b -P render/render.py -- \
        --object "$input_file" \
        --output_folder "$output_base/render_tex" \
        --resolution 512
    
    # Verify render_tex output
    if [ ! -f "$output_base/render_tex/000.png" ]; then
        echo "  ✗ WARNING: render_tex/000.png not found!"
    else
        echo "  ✓ render_tex completed"
    fi
    
    # Render render_cond folder with different lighting conditions
    echo "  → Rendering lighting conditions to render_cond..."
    $BLENDER_PATH -b -P render/render.py -- \
        --object "$input_file" \
        --output_folder "$output_base/render_cond" \
        --resolution 512 \
        --lighting_conditions "PL,AL,ENVMAP"
    
    # Verify render_cond output
    if [ ! -f "$output_base/render_cond/000_light_PL.png" ]; then
        echo "  ✗ WARNING: render_cond/000_light_PL.png not found!"
    else
        echo "  ✓ render_cond completed"
    fi
    
    # Generate watertight mesh and sample points
    # Note: mesh.ply should already exist from TRELLIS processing
    echo "  → Generating watertight mesh and sampling points..."
    if [ -f "$output_base/render_cond/mesh.ply" ]; then
        python3 watertight/watertight_and_sample.py \
            --input_obj "$output_base/render_cond/mesh.ply" \
            --output_prefix "$output_base/geo_data/$sha256"
        
        if [ -f "$output_base/geo_data/${sha256}_watertight.obj" ]; then
            echo "  ✓ geo_data completed"
        else
            echo "  ✗ WARNING: geo_data generation may have failed"
        fi
    else
        echo "  ✗ WARNING: mesh.ply not found in render_cond! Skipping watertight generation."
        echo "  (Expected from TRELLIS processing)"
    fi
    
    echo "  ✓ Completed: $file_identifier"
    return 0
}

# Read metadata.csv and process each row
# Expected CSV format: file_identifier,sha256,... (or other columns)
# We'll use Python to parse CSV properly
echo "Reading metadata.csv and processing files..."
echo ""

# Use Python to parse CSV and process files
python3 << EOF
import csv
import sys
import os
import subprocess
from datetime import datetime

# Try to import tqdm, fallback to simple progress if not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: tqdm not available, using simple progress display")

metadata_csv = "$METADATA_CSV"
raw_data_base_dir = "$RAW_DATA_BASE_DIR"
output_base_dir = "$OUTPUT_BASE_DIR"
blender_path = "$BLENDER_PATH"
log_file = "$LOG_FILE"

# Log file handler
log_fp = open(log_file, 'a', encoding='utf-8')

def log(message, to_console=False):
    """Write to log file and optionally to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    log_fp.write(log_message + "\n")
    log_fp.flush()  # Ensure real-time writing
    if to_console:
        print(message)

def process_file(file_identifier, sha256, local_path=None):
    """Process a single file"""
    # Determine input file path
    if local_path:
        # Use local_path if provided
        if os.path.isabs(local_path):
            # Absolute path: use as-is
            input_file = local_path
        else:
            # Relative path: join with raw_data_base_dir
            input_file = os.path.join(raw_data_base_dir, local_path)
    else:
        # Fallback: use file_identifier (for backward compatibility)
        input_file = os.path.join(raw_data_base_dir, file_identifier)
    
    output_base = os.path.join(output_base_dir, sha256)
    
    if not os.path.exists(input_file):
        display_name = local_path if local_path else file_identifier
        log(f"✗ SKIP {display_name}: File not found: {input_file}")
        return False
    
    display_name = local_path if local_path else file_identifier
    log(f"Processing: {display_name}")
    log(f"  SHA256: {sha256}")
    log(f"  Input: {input_file}")
    log(f"  Output: {output_base}")
    
    # Create output directories
    os.makedirs(os.path.join(output_base, "render_cond"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "render_tex"), exist_ok=True)
    os.makedirs(os.path.join(output_base, "geo_data"), exist_ok=True)
    
    # Render render_tex
    log("  → Rendering PBR materials to render_tex...")
    cmd = [
        blender_path, "-b", "-P", "render/render.py", "--",
        "--object", input_file,
        "--output_folder", os.path.join(output_base, "render_tex"),
        "--resolution", "512"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ✗ ERROR in render_tex: {result.stderr}")
        if result.stdout:
            log(f"  STDOUT: {result.stdout}")
        return False
    if result.stdout:
        log(f"  render_tex stdout: {result.stdout}")
    
    if not os.path.exists(os.path.join(output_base, "render_tex", "000.png")):
        log(f"  ✗ WARNING: render_tex/000.png not found!")
    else:
        log(f"  ✓ render_tex completed")
    
    # Render render_cond
    log("  → Rendering lighting conditions to render_cond...")
    cmd = [
        blender_path, "-b", "-P", "render/render.py", "--",
        "--object", input_file,
        "--output_folder", os.path.join(output_base, "render_cond"),
        "--resolution", "512",
        "--lighting_conditions", "PL,AL,ENVMAP"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"  ✗ ERROR in render_cond: {result.stderr}")
        if result.stdout:
            log(f"  STDOUT: {result.stdout}")
        return False
    if result.stdout:
        log(f"  render_cond stdout: {result.stdout}")
    
    if not os.path.exists(os.path.join(output_base, "render_cond", "000_light_PL.png")):
        log(f"  ✗ WARNING: render_cond/000_light_PL.png not found!")
    else:
        log(f"  ✓ render_cond completed")
    
    # Generate watertight mesh
    log("  → Generating watertight mesh and sampling points...")
    mesh_ply = os.path.join(output_base, "render_cond", "mesh.ply")
    if os.path.exists(mesh_ply):
        cmd = [
            "python3", "watertight/watertight_and_sample.py",
            "--input_obj", mesh_ply,
            "--output_prefix", os.path.join(output_base, "geo_data", sha256)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log(f"  ✗ ERROR in geo_data: {result.stderr}")
            if result.stdout:
                log(f"  STDOUT: {result.stdout}")
            return False
        if result.stdout:
            log(f"  geo_data stdout: {result.stdout}")
        
        if os.path.exists(os.path.join(output_base, "geo_data", f"{sha256}_watertight.obj")):
            log(f"  ✓ geo_data completed")
        else:
            log(f"  ✗ WARNING: geo_data generation may have failed")
    else:
        log(f"  ✗ WARNING: mesh.ply not found in render_cond! Skipping watertight generation.")
        log(f"  (Expected from TRELLIS processing)")
    
    log(f"  ✓ Completed: {display_name}")
    return True

# Read and process CSV
try:
    with open(metadata_csv, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        first_line = f.readline()
        f.seek(0)
        
        if ',' in first_line:
            delimiter = ','
        elif '\t' in first_line:
            delimiter = '\t'
        else:
            delimiter = ','
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Find file_identifier, sha256, and local_path columns (case-insensitive)
        fieldnames = [f.lower() for f in reader.fieldnames]
        file_id_col = None
        sha256_col = None
        local_path_col = None
        
        for i, field in enumerate(reader.fieldnames):
            field_lower = field.lower()
            if 'file_identifier' in field_lower or 'fileidentifier' in field_lower or 'file_id' in field_lower:
                file_id_col = field
            if 'sha256' in field_lower or 'sha' in field_lower:
                sha256_col = field
            if 'local_path' in field_lower or 'localpath' in field_lower:
                local_path_col = field
        
        if file_id_col is None:
            error_msg = f"ERROR: Could not find 'file_identifier' column in CSV\nAvailable columns: {', '.join(reader.fieldnames)}"
            log(error_msg, to_console=True)
            sys.exit(1)
        
        if sha256_col is None:
            error_msg = f"ERROR: Could not find 'sha256' column in CSV\nAvailable columns: {', '.join(reader.fieldnames)}"
            log(error_msg, to_console=True)
            sys.exit(1)
        
        log(f"Using columns:", to_console=True)
        log(f"  file_identifier: {file_id_col}", to_console=True)
        log(f"  sha256: {sha256_col}", to_console=True)
        if local_path_col:
            log(f"  local_path: {local_path_col} (will be used for input file path)", to_console=True)
        else:
            log(f"  local_path: not found (will use file_identifier as fallback)", to_console=True)
        log("", to_console=True)
        
        # Read all rows first to get total count
        rows = list(reader)
        total = len(rows)
        
        if total == 0:
            log("ERROR: No rows found in CSV", to_console=True)
            sys.exit(1)
        
        log(f"Found {total} files to process", to_console=True)
        log("", to_console=True)
        
        # Process each row with progress bar
        success = 0
        failed = 0
        
        if HAS_TQDM:
            pbar = tqdm(rows, desc="Processing", unit="file", ncols=100)
        else:
            pbar = rows
        
        for idx, row in enumerate(pbar, 1):
            file_identifier = row[file_id_col].strip() if file_id_col in row else ""
            sha256 = row[sha256_col].strip() if sha256_col in row else ""
            local_path = row[local_path_col].strip() if local_path_col and local_path_col in row else None
            
            if not sha256:
                log(f"✗ SKIP row {idx}: Missing sha256")
                failed += 1
                if HAS_TQDM:
                    pbar.set_postfix({"Success": success, "Failed": failed})
                continue
            
            if not file_identifier and not local_path:
                log(f"✗ SKIP row {idx}: Missing both file_identifier and local_path")
                failed += 1
                if HAS_TQDM:
                    pbar.set_postfix({"Success": success, "Failed": failed})
                continue
            
            # Update progress bar description
            display_name = local_path if local_path else file_identifier
            if HAS_TQDM:
                pbar.set_description(f"Processing [{idx}/{total}]")
                pbar.set_postfix({"Current": os.path.basename(display_name)[:30]})
            
            log(f"----------------------------------------")
            log(f"[{idx}/{total}] Processing: {display_name}")
            log(f"  SHA256: {sha256}")
            if local_path:
                log(f"  Using local_path: {local_path}")
            
            if process_file(file_identifier, sha256, local_path=local_path):
                success += 1
                if HAS_TQDM:
                    pbar.set_postfix({"Success": success, "Failed": failed, "Current": "✓"})
            else:
                failed += 1
                if HAS_TQDM:
                    pbar.set_postfix({"Success": success, "Failed": failed, "Current": "✗"})
            
            log("")
        
        log("=" * 50, to_console=True)
        log(f"Summary:", to_console=True)
        log(f"  Total: {total}", to_console=True)
        log(f"  Success: {success}", to_console=True)
        log(f"  Failed: {failed}", to_console=True)
        log("=" * 50, to_console=True)

except Exception as e:
    error_msg = f"ERROR processing CSV: {e}"
    log(error_msg, to_console=True)
    import traceback
    traceback_msg = traceback.format_exc()
    log(traceback_msg)
    sys.exit(1)
finally:
    log_fp.close()
EOF

echo ""
echo "=== Batch Rendering Complete ==="
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Expected structure:"
echo "  $OUTPUT_BASE_DIR"
echo "  ├── {sha2561}/"
echo "  │   ├── render_tex/"
echo "  │   ├── render_cond/"
echo "  │   └── geo_data/"
echo "  ├── {sha2562}/"
echo "  │   └── ..."
echo "  └── ..."
