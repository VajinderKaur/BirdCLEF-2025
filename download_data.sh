#!/bin/bash

# Script to download BirdCLEF-2025 competition data from Kaggle
# Make sure you have the Kaggle API credentials configured

# Check if kaggle.json exists
KAGGLE_CONFIG_DIR=~/.kaggle
KAGGLE_CONFIG_FILE=$KAGGLE_CONFIG_DIR/kaggle.json

if [ ! -f "$KAGGLE_CONFIG_FILE" ]; then
    echo "Error: Kaggle API credentials not found at $KAGGLE_CONFIG_FILE"
    echo "Please follow these steps to set up your Kaggle API credentials:"
    echo "1. Login to https://www.kaggle.com/"
    echo "2. Go to 'Account' section"
    echo "3. Scroll down to 'API' section and click 'Create New API Token'"
    echo "4. This will download a kaggle.json file"
    echo "5. Create the directory with: mkdir -p ~/.kaggle"
    echo "6. Move the downloaded file: mv /path/to/downloaded/kaggle.json ~/.kaggle/"
    echo "7. Set the correct permissions: chmod 600 ~/.kaggle/kaggle.json"
    echo "8. Run this script again"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Navigate to data directory
cd data

echo "Downloading BirdCLEF-2025 data... zip file will be available in the data/ directory once finished"

# Download data using Kaggle CLI
kaggle competitions download -c birdclef-2025

# Extract the downloaded zip file
if [ -f birdclef-2025.zip ]; then
    echo "Extracting birdclef-2025.zip..."
    unzip -q birdclef-2025.zip
    
    # Optionally remove the zip file
    echo "Removing birdclef-2025.zip..."
    rm birdclef-2025.zip
    
    echo "Download and extraction complete!"
else
    echo "Error: birdclef-2025.zip not found after download."
    exit 1
fi 