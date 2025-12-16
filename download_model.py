#!/usr/bin/env python3
"""
Download script for NeuroScan AI model files.
Run this script to download the pre-trained model.
"""

import os
import requests
from pathlib import Path

def download_model():
    """Download the pre-trained model file."""
    
    # Model download URL (you'll need to host this somewhere)
    # For now, this is a placeholder
    model_url = "https://github.com/AviMath2412/NeuroScan-AI/releases/download/v1.0/best_model.h5"
    model_filename = "best_model.h5"
    
    print("🧠 NeuroScan AI - Model Download")
    print("=" * 40)
    
    if os.path.exists(model_filename):
        print(f"✅ Model file '{model_filename}' already exists!")
        return
    
    print(f"📥 Downloading model from: {model_url}")
    print("⏳ This may take a few minutes...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📊 Progress: {progress:.1f}%", end="", flush=True)
        
        print(f"\n✅ Model downloaded successfully: {model_filename}")
        print(f"📁 File size: {os.path.getsize(model_filename) / (1024*1024):.1f} MB")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Download failed: {e}")
        print("\n📝 Manual download instructions:")
        print("1. Go to: https://github.com/AviMath2412/NeuroScan-AI/releases")
        print("2. Download the model file")
        print(f"3. Place it as '{model_filename}' in the project root")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    download_model()