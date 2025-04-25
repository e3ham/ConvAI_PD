#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import torchaudio
from tqdm import tqdm
import argparse

def get_audio_duration(file_path):
    """Get the exact duration of an audio file in seconds."""
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 40.0  # Default duration

def create_comprehensive_manifest(original_manifest_path, data_dir, output_manifest_path, include_augmented=True):
    """
    Create a comprehensive manifest that includes all WAV files in the directory.
    
    Args:
        original_manifest_path: Path to the original manifest JSON file
        data_dir: Root directory containing audio files
        output_manifest_path: Path to save the new manifest
        include_augmented: Whether to include augmented files
    """
    print(f"Creating comprehensive manifest from {original_manifest_path}")
    print(f"Data directory: {data_dir}")
    
    # Load the original manifest
    with open(original_manifest_path, 'r') as f:
        original_manifest = json.load(f)
    
    # Create a dictionary to map filenames to metadata
    filename_to_metadata = {}
    
    # Process original manifest entries
    for key, info in original_manifest.items():
        # Get the filename part of the path
        path = info["path"]
        filename = os.path.basename(path)
        
        # Store the metadata for this file
        filename_to_metadata[filename] = {
            "key": key,
            "info": info
        }
    
    print(f"Original manifest has {len(original_manifest)} entries")
    
    # Scan the data directory for all WAV files
    all_wav_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                # Include all files or only augmented if requested
                if include_augmented or "_aug" not in file:
                    all_wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_wav_files)} WAV files in {data_dir}")
    
    # Create a new manifest with all WAV files
    new_manifest = {}
    
    # First, add all entries from the original manifest
    new_manifest.update(original_manifest)
    
    # Track statistics
    original_count = len(new_manifest)
    added_count = 0
    
    # Process any WAV files that aren't in the original manifest
    for wav_path in tqdm(all_wav_files, desc="Processing WAV files"):
        filename = os.path.basename(wav_path)
        
        # Skip if this file is already in the new manifest
        skip = False
        for key, info in new_manifest.items():
            existing_filename = os.path.basename(info["path"])
            if existing_filename == filename:
                skip = True
                break
        
        if skip:
            continue
        
        # If this is an augmented file, find its original
        is_augmented = "_aug" in filename
        original_filename = None
        
        if is_augmented:
            # Extract the original filename by removing _aug1, _aug2, etc.
            original_filename = filename.replace("_aug1.wav", ".wav").replace("_aug2.wav", ".wav")
            
            # Check if we know about the original file
            if original_filename in filename_to_metadata:
                # Create metadata based on the original
                original_info = filename_to_metadata[original_filename]["info"]
                original_key = filename_to_metadata[original_filename]["key"]
                
                # Create a new key for the augmented file
                aug_identifier = "aug1" if "_aug1" in filename else "aug2"
                new_key = f"{original_key.split('.')[0]}_{aug_identifier}.wav"
                
                # Create the path based on the actual location
                relative_path = os.path.relpath(wav_path, data_dir)
                path_for_manifest = os.path.join("pd_dataset", relative_path).replace("\\", "/")
                
                # Get actual duration
                duration = get_audio_duration(wav_path)
                
                # Add to the new manifest
                new_manifest[new_key] = {
                    "path": path_for_manifest,
                    "group": original_info["group"],
                    "sex": original_info["sex"],
                    "age": original_info["age"],
                    "label": original_info["label"],
                    "duration": duration,
                    "augmented": True
                }
                
                added_count += 1
            else:
                print(f"Warning: Found augmented file without original: {filename}")
        else:
            # This is a new original file not in the manifest
            # We'll need to infer metadata from the path
            
            # Extract group from path
            group = "Unknown"
            if "Young Healthy Control" in wav_path:
                group = "15 Young Healthy Control"
            elif "Elderly Healthy Control" in wav_path:
                group = "22 Elderly Healthy Control"
            elif "People with Parkinson" in wav_path:
                group = "28 People with Parkinson's disease"
            
            # Extract speaker name (parent directory)
            speaker = os.path.basename(os.path.dirname(wav_path))
            
            # Infer label from group
            label = "HC"
            if "Parkinson" in group:
                label = "PD"
            
            # Extract sex and age if possible from filename
            sex = "M"  # Default
            age = 50   # Default
            
            # Try to extract from filename (common format: B1LBULCAAS94M100120171021)
            for i in range(len(filename) - 2):
                if i+2 < len(filename) and filename[i:i+2].isdigit():
                    if filename[i+2] in ["M", "F"]:
                        age = int(filename[i:i+2])
                        sex = filename[i+2]
                        break
            
            # Create a new key based on speaker and filename
            speaker_formatted = speaker.replace(" ", "_").upper()
            group_formatted = "_".join(group.split()[1:]).upper()
            new_key = f"{speaker_formatted}_{group_formatted}_{filename}"
            
            # Get actual duration
            duration = get_audio_duration(wav_path)
            
            # Create a relative path for the manifest
            relative_path = os.path.relpath(wav_path, data_dir)
            path_for_manifest = os.path.join("pd_dataset", relative_path).replace("\\", "/")
            
            # Add to the new manifest
            new_manifest[new_key] = {
                "path": path_for_manifest,
                "group": group,
                "sex": sex,
                "age": age,
                "label": label,
                "duration": duration
            }
            
            added_count += 1
    
    # Save the new manifest
    os.makedirs(os.path.dirname(output_manifest_path), exist_ok=True)
    with open(output_manifest_path, 'w') as f:
        json.dump(new_manifest, f, indent=2)
    
    print(f"Original entries: {original_count}")
    print(f"Added entries: {added_count}")
    print(f"Total entries in new manifest: {len(new_manifest)}")
    print(f"Saved to: {output_manifest_path}")
    
    return new_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create comprehensive manifest including all WAV files")
    parser.add_argument("--original_manifest", required=True, help="Path to the original manifest JSON file")
    parser.add_argument("--data_dir", required=True, help="Directory containing the audio files")
    parser.add_argument("--output_manifest", required=True, help="Path to save the new manifest")
    parser.add_argument("--include_augmented", action="store_true", help="Include augmented files in the manifest")
    
    args = parser.parse_args()
    
    create_comprehensive_manifest(
        args.original_manifest,
        args.data_dir,
        args.output_manifest,
        args.include_augmented
    )