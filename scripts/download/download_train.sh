#!/bin/bash
set -e

# === Configuration ===

# Default values
DEFAULT_START=0
DEFAULT_END=10
DEFAULT_DIR="./download/"

# Parse arguments
TARGET_DIR="${1:-$DEFAULT_DIR}"
START_INDEX="${2:-$DEFAULT_START}"
# END_INDEX="${3:-$DEFAULT_END}"

# Create the directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# URL components
BASE_URL="https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
PREFIX="v1.0-trainval"
SUFFIX="_blobs.tgz"

echo "Downloading blobs from ${START_INDEX} to ${END_INDEX} into ${TARGET_DIR}..."

i=$(printf "%02d" "$START_INDEX")

# Download loop
FILE="${PREFIX}${i}${SUFFIX}"
URL="${BASE_URL}/${FILE}"
DEST="${TARGET_DIR}/${FILE}"
echo "Downloading ${FILE}..."
wget -c "${URL}" -O "${DEST}"
echo "Saved to ${DEST}"

echo "✅ All downloads completed successfully to ${TARGET_DIR}."
