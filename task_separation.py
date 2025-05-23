import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def separate_tasks(data_dir, output_base_dir, manifests_dir=None, use_symlinks=True):
    """
    Separate audio files into different tasks (reading vs other) while preserving directory structure.
    
    This function scans the dataset directory for WAV files, determines if each file is a reading task
    (based on filename prefixes B1 or B2), and organizes them into separate task directories.
    It can either create symbolic links or copy the files to the new locations.
    
    Args:
        data_dir (str): Root directory of the original dataset containing audio files.
        output_base_dir (str): Base directory where separated task folders will be created.
        manifests_dir (str, optional): Directory containing manifest files for additional 
                                     processing. Defaults to None.
        use_symlinks (bool, optional): Whether to create symbolic links (True) or copy 
                                     files (False). Defaults to True.
    
    Returns:
        tuple: A pair of strings (reading_task_dir, other_task_dir) with paths to the 
              created task directories.
    
    Example:
        >>> reading_dir, other_dir = separate_tasks(
        ...     "dataset/audio",
        ...     "dataset/separated",
        ...     "dataset/manifests",
        ...     use_symlinks=True
        ... )
        Found 523 WAV files in dataset/audio
        Separating tasks: 100%|██████| 523/523 [00:03<00:00]
        
        Task separation complete!
        Reading task files:
         Original: 156
         Augmented: 312
         Total: 468
        
        Other task files:
         Original: 185
         Augmented: 370
         Total: 555
    """
    # Create output directories
    reading_task_dir = os.path.join(output_base_dir, "reading_task")
    other_task_dir = os.path.join(output_base_dir, "other_task")

    os.makedirs(reading_task_dir, exist_ok=True)
    os.makedirs(other_task_dir, exist_ok=True)

    # Track statistics
    stats = {
        "reading": {"original": 0, "augmented": 0},
        "other": {"original": 0, "augmented": 0}
    }

    # Scan all WAV files in the dataset
    all_wav_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                all_wav_files.append(os.path.join(root, file))

    print(f"Found {len(all_wav_files)} WAV files in {data_dir}")

    # Process each WAV file
    for wav_path in tqdm(all_wav_files, desc="Separating tasks"):
        # Get the path relative to the data directory
        rel_path = os.path.relpath(wav_path, data_dir)

        # Determine which task folder to use
        filename = os.path.basename(wav_path)

        # Check if it's a reading task (starts with B1 or B2)
        is_reading = filename.startswith("B1") or filename.startswith("B2")

        # Determine if it's an augmented file
        is_augmented = "_aug" in filename

        # Set target directory based on task
        if is_reading:
            target_dir = reading_task_dir
            if is_augmented:
                stats["reading"]["augmented"] += 1
            else:
                stats["reading"]["original"] += 1
        else:
            target_dir = other_task_dir
            if is_augmented:
                stats["other"]["augmented"] += 1
            else:
                stats["other"]["original"] += 1

        # Create full target path
        target_path = os.path.join(target_dir, rel_path)

        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Create symlink or copy file
        if os.path.exists(target_path):
            # Skip if file already exists
            continue

        if use_symlinks:
            try:
                # Use absolute path for the source
                abs_source = os.path.abspath(wav_path)
                os.symlink(abs_source, target_path)
            except Exception as e:
                print(f"Error creating symlink for {wav_path}: {e}")
                # Fall back to copying
                shutil.copy2(wav_path, target_path)
        else:
            try:
                shutil.copy2(wav_path, target_path)
            except Exception as e:
                print(f"Error copying {wav_path}: {e}")

    # Print statistics
    print("\nTask separation complete!")
    print("Reading task files:")
    print(f"  Original: {stats['reading']['original']}")
    print(f"  Augmented: {stats['reading']['augmented']}")
    print(
        f"  Total: {stats['reading']['original'] + stats['reading']['augmented']}")

    print("\nOther task files:")
    print(f"  Original: {stats['other']['original']}")
    print(f"  Augmented: {stats['other']['augmented']}")
    print(
        f"  Total: {stats['other']['original'] + stats['other']['augmented']}")

    # Create manifests for each task if manifest directory is provided
    if manifests_dir and os.path.exists(manifests_dir):
        create_task_manifests(
            manifests_dir,
            data_dir,
            reading_task_dir,
            other_task_dir,
            output_base_dir)

    return reading_task_dir, other_task_dir


def create_task_manifests(manifests_dir, original_data_dir, reading_task_dir, 
                         other_task_dir, output_base_dir):
    """
    Create separate manifests for each task based on original manifests.
    
    This function processes the original manifest files (train, valid, test) and creates
    separate manifests for reading and other tasks, updating the file paths to point
    to the appropriate task directories.
    
    Args:
        manifests_dir (str): Directory containing original manifest files.
        original_data_dir (str): Original data directory path for relative path calculations.
        reading_task_dir (str): Directory containing reading task files.
        other_task_dir (str): Directory containing other task files.
        output_base_dir (str): Base directory where output manifest directories will be created.
    
    Returns:
        None: This function creates files on disk but does not return any values.
    
    Example:
        >>> create_task_manifests(
        ...     "dataset/manifests",
        ...     "dataset/audio",
        ...     "dataset/separated/reading_task",
        ...     "dataset/separated/other_task",
        ...     "dataset/separated"
        ... )
        Processing train manifest: 100%|██████| 341/341 [00:00<00:00]
        Created train manifests:
         Reading task: 156 entries
         Other task: 185 entries
        Processing valid manifest: 100%|██████| 72/72 [00:00<00:00]
        Created valid manifests:
         Reading task: 31 entries
         Other task: 41 entries
        Processing test manifest: 100%|██████| 110/110 [00:00<00:00]
        Created test manifests:
         Reading task: 49 entries
         Other task: 61 entries
    """
    # Create output directories for manifests
    reading_manifests_dir = os.path.join(output_base_dir, "reading_manifests")
    other_manifests_dir = os.path.join(output_base_dir, "other_manifests")

    os.makedirs(reading_manifests_dir, exist_ok=True)
    os.makedirs(other_manifests_dir, exist_ok=True)

    # Process each split (train, valid, test)
    for split in ["train", "valid", "test"]:
        manifest_path = os.path.join(manifests_dir, f"{split}.json")

        # Skip if manifest doesn't exist
        if not os.path.exists(manifest_path):
            print(f"Manifest not found: {manifest_path}")
            continue

        # Load original manifest
        with open(manifest_path, 'r') as f:
            original_manifest = json.load(f)

        # Create separate manifests for each task
        reading_manifest = {}
        other_manifest = {}

        # Process each entry in the original manifest
        for key, info in tqdm(original_manifest.items(),
                              desc=f"Processing {split} manifest"):
            # Get the filename from the path
            file_path = info["path"]
            filename = os.path.basename(file_path)

            # Check if it's a reading task
            is_reading = filename.startswith("B1") or filename.startswith("B2")

            # Update paths to point to the new directories
            if is_reading:
                # Create updated info with path pointing to reading task
                # directory
                updated_info = dict(info)

                # Adjust the path if needed
                if not os.path.exists(
                        file_path) and original_data_dir in file_path:
                    # If original path is absolute and contains the original
                    # data dir
                    rel_path = os.path.relpath(file_path, original_data_dir)
                    updated_info["path"] = os.path.join(
                        reading_task_dir, rel_path)
                elif not os.path.exists(file_path) and not os.path.isabs(file_path):
                    # If original path is relative
                    updated_info["path"] = os.path.join(
                        reading_task_dir, file_path)

                # Add to reading manifest
                reading_manifest[key] = updated_info
            else:
                # Create updated info with path pointing to other task
                # directory
                updated_info = dict(info)

                # Adjust the path if needed
                if not os.path.exists(
                        file_path) and original_data_dir in file_path:
                    # If original path is absolute and contains the original
                    # data dir
                    rel_path = os.path.relpath(file_path, original_data_dir)
                    updated_info["path"] = os.path.join(
                        other_task_dir, rel_path)
                elif not os.path.exists(file_path) and not os.path.isabs(file_path):
                    # If original path is relative
                    updated_info["path"] = os.path.join(
                        other_task_dir, file_path)

                # Add to other manifest
                other_manifest[key] = updated_info

        # Save reading manifest
        reading_manifest_path = os.path.join(
            reading_manifests_dir, f"{split}.json")
        with open(reading_manifest_path, 'w') as f:
            json.dump(reading_manifest, f, indent=2)

        # Save other manifest
        other_manifest_path = os.path.join(
            other_manifests_dir, f"{split}.json")
        with open(other_manifest_path, 'w') as f:
            json.dump(other_manifest, f, indent=2)

        print(f"Created {split} manifests:")
        print(f"  Reading task: {len(reading_manifest)} entries")
        print(f"  Other task: {len(other_manifest)} entries")


def scan_augmented_files(data_dir, augmented_dir, output_base_dir):
    """
    Scan and organize augmented files into appropriate task directories.
    
    This function finds all augmented WAV files in the specified directory, determines
    their task type based on filename, and creates symbolic links or copies them to
    the corresponding task directory while preserving their relative paths.
    
    Args:
        data_dir (str): Original data directory for path reference.
        augmented_dir (str): Directory containing augmented audio files.
        output_base_dir (str): Base directory containing the task directories.
    
    Returns:
        None: This function creates files on disk but does not return any values.
    
    Example:
        >>> scan_augmented_files(
        ...     "dataset/audio",
        ...     "dataset/augmented",
        ...     "dataset/separated"
        ... )
        Found 682 augmented WAV files
        Organizing augmented files: 100%|██████| 682/682 [00:02<00:00]
    """
    # Create task directories
    reading_task_dir = os.path.join(output_base_dir, "reading_task")
    other_task_dir = os.path.join(output_base_dir, "other_task")

    os.makedirs(reading_task_dir, exist_ok=True)
    os.makedirs(other_task_dir, exist_ok=True)

    # Find all augmented WAV files
    augmented_files = []
    for root, _, files in os.walk(augmented_dir):
        for file in files:
            if file.endswith(".wav") and ("_aug" in file):
                augmented_files.append(os.path.join(root, file))

    print(f"Found {len(augmented_files)} augmented WAV files")

    # Process each augmented file
    for aug_path in tqdm(augmented_files, desc="Organizing augmented files"):
        # Get filename
        filename = os.path.basename(aug_path)

        # Determine if it's a reading task
        is_reading = filename.startswith("B1") or filename.startswith("B2")

        # Get relative path within augmented directory
        try:
            rel_path = os.path.relpath(aug_path, augmented_dir)
        except ValueError:
            # Path is not relative to augmented_dir
            rel_path = os.path.basename(aug_path)

        # Determine target directory
        target_dir = reading_task_dir if is_reading else other_task_dir

        # Create target path
        target_path = os.path.join(target_dir, rel_path)

        # Create parent directories
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        # Create symlink or copy
        if not os.path.exists(target_path):
            try:
                # Use absolute path for the source
                abs_source = os.path.abspath(aug_path)
                os.symlink(abs_source, target_path)
            except Exception as e:
                try:
                    shutil.copy2(aug_path, target_path)
                except Exception as e2:
                    print(f"Error copying {aug_path}: {e2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate audio files into different tasks")

    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root directory of the original dataset")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base directory where separated task folders will be created")
    parser.add_argument(
        "--manifests_dir",
        help="Directory containing manifest files (optional)")
    parser.add_argument(
        "--augmented_dir",
        help="Directory containing augmented files (optional)")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks")

    args = parser.parse_args()

    # Run the task separation
    reading_dir, other_dir = separate_tasks(
        args.data_dir,
        args.output_dir,
        args.manifests_dir,
        use_symlinks=not args.copy
    )

    # Handle augmented files if specified
    if args.augmented_dir and os.path.exists(args.augmented_dir):
        scan_augmented_files(
            args.data_dir,
            args.augmented_dir,
            args.output_dir)
