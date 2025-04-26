#!/usr/bin/env python3
import os
import json
import glob
import torchaudio
from pathlib import Path
from tqdm import tqdm


def get_audio_duration(file_path):
    """Get the exact duration of an audio file in seconds.
    
    Args:
        file_path (str): Path to the audio file to analyze.
        
    Returns:
        float: Duration of the audio file in seconds.
        
    Example:
        >>> duration = get_audio_duration("recording.wav")
        >>> print(f"The recording is {duration:.2f} seconds long")
        The recording is 5.32 seconds long
    """
    try:
        info = torchaudio.info(file_path)
        duration = info.num_frames / info.sample_rate
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")


def create_augmented_manifest(
        manifest_path,
        augmented_dir,
        output_path,
        augmentation_factor=2):
    """
    Create a new manifest that includes both original and augmented audio files.
    
    This function loads an original manifest file, finds all corresponding 
    augmented audio files in the specified directory, and creates a new manifest
    that includes both original and augmented entries with appropriate metadata.
    The function handles various file naming patterns and attempts to match
    augmented files with their original counterparts.
    
    Args:
        manifest_path (str): Path to the original manifest JSON file.
        augmented_dir (str): Directory containing augmented audio files.
        output_path (str): Path where the new augmented manifest will be saved.
        augmentation_factor (int, optional): Expected number of augmentations 
            per original file. Defaults to 2.
    
    Returns:
        dict: The complete augmented manifest data structure that was saved to disk.
            Each entry contains metadata including path, group, sex, age, label,
            duration, and augmentation status.
    
    Example:
        >>> original_manifest = "manifests/train.json"
        >>> augmented_dir = "augmented_data/"
        >>> output_path = "aug_manifests/train_with_augmentation.json"
        >>> augmented_data = create_augmented_manifest(
        ...     original_manifest, 
        ...     augmented_dir, 
        ...     output_path
        ... )
        Creating augmented manifest from manifests/train.json
        Added 150 original entries
        Found 450 total WAV files in augmented directory
        Of which 300 are augmented files
        Processing manifest entries: 100%|██████| 150/150 [00:05<00:00]
        Added 300 augmented entries linked to original files
        Added 0 additional augmented entries
        Skipped 0 missing augmentations
        Total manifest entries: 450
        Saved augmented manifest to aug_manifests/train_with_augmentation.json
        >>> print(f"Original entries: {len(original_data)}")
        >>> print(f"Augmented entries: {len(augmented_data) - len(original_data)}")
        Original entries: 150
        Augmented entries: 300
    """
    print(f"Creating augmented manifest from {manifest_path}")

    # Load the original manifest
    with open(manifest_path, 'r') as f:
        original_manifest = json.load(f)

    # Create a new manifest that will include both original and augmented files
    augmented_manifest = {}

    # First, add all original entries
    augmented_manifest.update(original_manifest)
    print(f"Added {len(original_manifest)} original entries")

    # First, scan the entire augmented directory to find all WAV files
    print("Scanning augmented directory for all WAV files...")
    all_wav_files = []
    for root, _, files in os.walk(augmented_dir):
        for file in files:
            if file.endswith(".wav"):
                all_wav_files.append(os.path.join(root, file))

    print(f"Found {len(all_wav_files)} total WAV files in augmented directory")

    # Filter to find only augmented files
    augmented_files = [
        f for f in all_wav_files if "_aug" in os.path.basename(f)]
    print(f"Of which {len(augmented_files)} are augmented files")

    # Create a mapping from original filenames to lists of augmented files
    original_to_augmented = {}
    for aug_file in augmented_files:
        filename = os.path.basename(aug_file)
        original_filename = filename.replace(
            "_aug1.wav", ".wav").replace(
            "_aug2.wav", ".wav")

        if original_filename not in original_to_augmented:
            original_to_augmented[original_filename] = []

        original_to_augmented[original_filename].append(aug_file)

    # Track statistics
    augmented_added = 0
    augmented_skipped = 0

    # For each original entry in the manifest
    for key, info in tqdm(original_manifest.items(),
                          desc="Processing manifest entries"):
        original_path = info["path"]
        filename = os.path.basename(original_path)

        # Check if we found any augmented versions of this file
        if filename in original_to_augmented:
            aug_files = original_to_augmented[filename]

            # Add each augmented file to the manifest
            for aug_file in aug_files:
                aug_filename = os.path.basename(aug_file)

                # Determine augmentation number
                aug_num = 1  # Default
                if "_aug1" in aug_filename:
                    aug_num = 1
                elif "_aug2" in aug_filename:
                    aug_num = 2
                elif "_aug" in aug_filename:
                    # Try to extract number from the filename
                    try:
                        parts = aug_filename.split("_aug")
                        aug_num = int(parts[1].split(".")[0])
                    except BaseException:
                        pass

                # Create a key for the augmented file
                aug_key = f"{key.split('.')[0]}_aug{aug_num}.wav"

                # Get duration of the augmented file
                duration = get_audio_duration(aug_file)

                # Add to manifest
                augmented_manifest[aug_key] = {
                    "path": aug_file,
                    "group": info["group"],
                    "sex": info["sex"],
                    "age": info["age"],
                    "label": info["label"],
                    "duration": duration,
                    "augmented": True
                }

                augmented_added += 1
        else:
            # No augmented versions found for this file
            augmented_skipped += len(original_to_augmented.get(filename, []))

    # Additionally, look for any augmented files that might not be linked to
    # original manifest entries
    extra_augmented_count = 0

    # Create a set of filenames already in the manifest
    existing_filenames = {
        os.path.basename(
            info["path"]) for info in augmented_manifest.values()}

    # Check each augmented file
    for aug_file in augmented_files:
        aug_filename = os.path.basename(aug_file)

        # Skip if this file is already included
        if aug_filename in existing_filenames:
            continue

        # Try to determine group and extract information from the path
        path_parts = Path(aug_file).parts

        # Extract group name
        group = "Unknown"
        for part in path_parts:
            if "Young Healthy Control" in part:
                group = "15 Young Healthy Control"
                break
            elif "Elderly Healthy Control" in part:
                group = "22 Elderly Healthy Control"
                break
            elif "Parkinson" in part:
                group = "28 People with Parkinson's disease"
                break

        # Extract speaker name (parent directory)
        speaker = os.path.basename(os.path.dirname(aug_file))

        # Determine label from group
        label = "HC"
        if "Parkinson" in group:
            label = "PD"

        # Try to extract sex and age from filename
        sex = "M"  # Default
        age = 50   # Default

        # Create a key for this augmented file
        aug_key = f"{speaker.replace(' ', '_').upper()}_{group.split()[-1].upper()}_{aug_filename}"

        # Get duration
        duration = get_audio_duration(aug_file)

        # Add to manifest
        augmented_manifest[aug_key] = {
            "path": aug_file,
            "group": group,
            "sex": sex,
            "age": age,
            "label": label,
            "duration": duration,
            "augmented": True
        }

        extra_augmented_count += 1

    print(f"Added {augmented_added} augmented entries linked to original files")
    print(f"Added {extra_augmented_count} additional augmented entries")
    print(f"Skipped {augmented_skipped} missing augmentations")
    print(f"Total manifest entries: {len(augmented_manifest)}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the new manifest
    with open(output_path, 'w') as f:
        json.dump(augmented_manifest, f, indent=2)

    print(f"Saved augmented manifest to {output_path}")
    return augmented_manifest


def update_all_manifests(manifest_dir, augmented_dir, output_dir):
    """
    Process all manifest files (train, valid, test) in a directory to include augmented data.
    
    This function looks for standard split manifest files (train.json, valid.json, test.json)
    in the specified directory and creates new manifests that include augmented audio data
    for each split. The new manifests are saved with "_with_augmentation" suffix.
    
    Args:
        manifest_dir (str): Directory containing original manifest files (train.json, 
                           valid.json, test.json).
        augmented_dir (str): Directory containing augmented audio files.
        output_dir (str): Directory where the new augmented manifest files will be saved.
    
    Returns:
        None: This function doesn't return any value but creates new manifest files
              on disk for each available split.
    
    Example:
        >>> update_all_manifests(
        ...     manifest_dir="manifests",
        ...     augmented_dir="augmented_data",
        ...     output_dir="aug_manifests"
        ... )
        Creating augmented manifest from manifests/train.json
        Added 150 original entries
        ...
        Saved augmented manifest to aug_manifests/train_with_augmentation.json
        
        Creating augmented manifest from manifests/valid.json
        Added 30 original entries
        ...
        Saved augmented manifest to aug_manifests/valid_with_augmentation.json
        
        Creating augmented manifest from manifests/test.json
        Added 50 original entries
        ...
        Saved augmented manifest to aug_manifests/test_with_augmentation.json
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "valid", "test"]:
        manifest_path = os.path.join(manifest_dir, f"{split}.json")
        if os.path.exists(manifest_path):
            output_path = os.path.join(
                output_dir, f"{split}_with_augmentation.json")
            create_augmented_manifest(
                manifest_path, augmented_dir, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create manifest files including augmented data")
    parser.add_argument(
        "--manifest_dir",
        required=True,
        help="Directory containing original manifests")
    parser.add_argument(
        "--augmented_dir",
        required=True,
        help="Directory containing augmented files")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save new manifests")

    args = parser.parse_args()

    update_all_manifests(
        args.manifest_dir,
        args.augmented_dir,
        args.output_dir)

# python aug.py \
#   --manifest_dir manifests \
#   --augmented_dir augmented_data \
#   --output_dir aug_manifests
