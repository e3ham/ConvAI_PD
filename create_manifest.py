import os
import json
import glob
import random
from pathlib import Path

import pandas as pd
import torchaudio
import numpy as np


def get_audio_duration(file_path):
    """Get the exact duration of an audio file in seconds.
    
    Args:
        file_path (str): Path to the audio file to analyze.
        
    Returns:
        float: Duration of the audio file in seconds.
        
    Example:
        >>> duration = get_audio_duration("recordings/patient001.wav")
        >>> print(f"The recording is {duration:.2f} seconds long")
        The recording is 5.32 seconds long
    """
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def read_group_metadata(group_dir_path):
    """Read and process metadata from Excel files in a group directory.
    
    This function scans a directory for Excel (.xlsx) files containing participant 
    metadata, normalizes column names, and creates a dictionary mapping participant 
    names to their metadata. It attempts multiple naming formats to increase 
    the chance of successful matching later.
    
    Args:
        group_dir_path (str): Path to the directory containing Excel metadata files.
        
    Returns:
        dict: A dictionary mapping participant names (in various formats) to their 
              metadata. Each metadata entry contains: sex, age, from (location), 
              time1/2/3, cps1/2/3 (characters per second).
        
    Example:
        >>> metadata = read_group_metadata("data/Young Healthy Control")
        >>> print(f"Loaded metadata for {len(metadata)} participants")
        >>> if "john s" in metadata:
        ...     print(f"John S. is {metadata['john s']['age']} years old")
        Loaded metadata for 52 participants
        John S. is 25 years old
    """
    person_metadata = {}
    for excel_file in glob.glob(os.path.join(group_dir_path, "*.xlsx")):
        file_name = os.path.basename(excel_file)
        if "FILE CODES" in file_name:  # Skip the FILE CODES.xlsx file
            continue

        print(f"  Reading metadata from {file_name}")
        try:
            # Use second row as header and normalize
            df = pd.read_excel(excel_file, engine="openpyxl", header=1)
            print(f"    DEBUG columns: {df.columns.tolist()}")
            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()
            # Make column names consistent
            if 'time 1' in df.columns:
                df.rename(columns={'time 1': 'time1'}, inplace=True)
            if 'time 2' in df.columns:
                df.rename(columns={'time 2': 'time2'}, inplace=True)
            if 'time 3' in df.columns:
                df.rename(columns={'time 3': 'time3'}, inplace=True)
        except Exception as e:
            print(f"    Error loading {file_name}: {e}")
            continue

        if df.empty or 'name' not in df.columns:
            print(f"    No usable data found in {file_name}")
            continue

        # Handle the case where surname column doesn't exist
        if 'surname' not in df.columns:
            print(
                f"    'surname' column not found in {file_name}, creating empty column")
            df['surname'] = ""

        for _, row in df.iterrows():
            if pd.isna(row['name']):
                continue

            # Clean the name field
            name = str(row['name']).strip()

            # Clean the surname field - handle [object Object] and other
            # oddities
            surname = str(row.get('surname', '')).strip()
            if surname.startswith('[object Object]') or pd.isna(
                    row.get('surname')):
                # Take the first letter from the folder name if available
                surname = ""

            # Create the uppercase format key that matches folder structure
            uppercase_key = f"{name} {surname}".strip().upper()
            if surname and len(surname) == 1:
                uppercase_key = f"{name} {surname}".upper()
            elif surname:
                uppercase_key = f"{name} {surname[0]}".upper()

            # Create the lowercase version for the dictionary
            lowercase_key = uppercase_key.lower()

            # Get age value, handling possible non-numeric values
            try:
                age = int(
                    row.get(
                        'age',
                        20)) if not pd.isna(
                    row.get('age')) else 20
            except BaseException:
                age = 20

            # Create metadata entry
            metadata_value = {
                'sex': row.get('sex', 'M'),
                'age': age,
                'from': row.get('from', ''),
                'time1': row.get('time1', 0),
                'cps1': row.get('cps1', 0) if 'cps1' in row else row.get('CPS1', 0),
                'time2': row.get('time2', 0),
                'cps2': row.get('cps2', 0) if 'cps2' in row else row.get('CPS2', 0),
                'time3': row.get('time3', 0),
                'cps3': row.get('cps3', 0) if 'cps3' in row else row.get('CPS3', 0),
            }

            # Store with multiple key formats to increase match chance
            person_metadata[lowercase_key] = metadata_value
            person_metadata[name.lower()] = metadata_value

            # Also store with name and first letter of surname (common format)
            if surname:
                person_metadata[f"{name.lower()} {surname[0].lower()}"] = metadata_value

            # Print what we're adding
            print(
                f"    Added metadata entry: '{uppercase_key}' -> {metadata_value['sex']}, {metadata_value['age']}")

        print(f"    Loaded metadata for {len(df)} rows from {file_name}")

    return person_metadata


def scan_pd_dataset(data_dir):
    """Scan a directory structure containing Parkinson's disease audio dataset.
    
    This function recursively scans a dataset directory containing three groups:
    young healthy controls, elderly healthy controls, and Parkinson's disease 
    patients. It finds all WAV files, attempts to match them with metadata, 
    and organizes them into a structured dictionary.
    
    Args:
        data_dir (str): Path to the root directory of the dataset.
        
    Returns:
        dict: A nested dictionary with the following structure:
            {
                "young_healthy": [
                    {"key": "PERSON_GROUP_file.wav", 
                     "info": {"path": "...", "group": "...", "sex": "...", 
                             "age": 25, "label": "HC", "duration": 5.4}},
                    ...
                ],
                "elderly_healthy": [...],
                "parkinsons": [...]
            }
        
    Example:
        >>> dataset = scan_pd_dataset("Italian_Parkinsons_Voice_and_Speech")
        >>> print(f"Dataset contains:")
        >>> for group, items in dataset.items():
        ...     print(f"  {group}: {len(items)} recordings")
        Dataset contains:
          young_healthy: 120 recordings
          elderly_healthy: 95 recordings
          parkinsons: 140 recordings
    """
    print(f"Scanning dataset at {data_dir}...")
    dataset = {"young_healthy": [], "elderly_healthy": [], "parkinsons": []}

    for group_dir in os.listdir(data_dir):
        group_path = os.path.join(data_dir, group_dir)
        if not os.path.isdir(group_path):
            continue

        # determine key
        if "Young Healthy" in group_dir:
            group_key = "young_healthy"
        elif "Elderly Healthy" in group_dir:
            group_key = "elderly_healthy"
        elif "Parkinson" in group_dir:
            group_key = "parkinsons"
        else:
            print(f"Skipping unknown folder: {group_dir}")
            continue

        print(f"Scanning group: {group_dir}")
        metadata_map = read_group_metadata(group_path)

        # Print out all keys in the metadata map
        print(f"  Available metadata keys ({len(metadata_map)}):")
        for key in sorted(list(metadata_map.keys())[:10]):
            print(f"    - '{key}'")
        if len(metadata_map) > 10:
            print(f"    - ...and {len(metadata_map)-10} more")

        # recursive find wavs
        wav_files = glob.glob(
            os.path.join(
                group_path,
                '**',
                '*.wav'),
            recursive=True)
        print(f"  Found {len(wav_files)} WAV files under {group_dir}")

        # Track missing metadata cases
        missing_metadata = set()

        for file_path in wav_files:
            rel_path = os.path.relpath(file_path, group_path)
            parts = Path(rel_path).parts
            person_dir = parts[0]  # First folder level is person name

            # For each file, we'll try multiple ways to look up metadata
            meta = None

            # 1. Try direct lowercase match
            lookup_key = person_dir.lower()
            if lookup_key in metadata_map:
                meta = metadata_map[lookup_key]

            # 2. Try to match by first name only
            if meta is None:
                name_parts = person_dir.split()
                if name_parts:
                    first_name = name_parts[0].lower()
                    if first_name in metadata_map:
                        meta = metadata_map[first_name]

            # 3. Try to match with the format "firstname initial"
            if meta is None and len(name_parts) >= 2:
                first_name = name_parts[0].lower()
                last_initial = name_parts[1][0].lower() if len(
                    name_parts[1]) > 0 else ""
                alt_key = f"{first_name} {last_initial}"
                if alt_key in metadata_map:
                    meta = metadata_map[alt_key]

            # If still no match, use default metadata and report the miss only
            # once
            if meta is None:
                if person_dir not in missing_metadata:
                    missing_metadata.add(person_dir)
                    print(
                        f"    No metadata match for '{person_dir}', tried: '{lookup_key}'")
                    if len(name_parts) >= 2:
                        print(
                            f"      Also tried: '{first_name}' and '{alt_key}'")

                # Parse person_dir for default metadata
                parts = person_dir.split(maxsplit=1)
                firstname = parts[0]
                lastname = parts[1] if len(parts) > 1 else ''
                meta = {'sex': 'M', 'age': 20, 'from': ''}

            duration = get_audio_duration(file_path)
            key = format_file_key(
                person_dir, group_dir, os.path.basename(file_path))
            label = "PD" if group_key == "parkinsons" else "HC"
            dataset[group_key].append({
                'key': key,
                'info': {
                    'path': file_path.replace(os.sep, '/'),
                    'group': group_dir,
                    'sex': meta['sex'],
                    'age': meta['age'],
                    'label': label,
                    'duration': duration
                }
            })

        # After processing all files in this group, summarize missing metadata
        if missing_metadata:
            print(
                f"  Missing metadata for {len(missing_metadata)} people in this group.")

    for k, v in dataset.items():
        print(f"  {k}: {len(v)} files")
    return dataset


def create_train_val_test_splits(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42):
    """Split the dataset into training, validation, and test sets.
    
    This function performs a person-based split (rather than a recording-based split),
    ensuring that all recordings from the same person stay in the same split.
    This avoids data leakage between splits.
    
    Args:
        dataset (dict): The dataset dictionary as returned by scan_pd_dataset().
        train_ratio (float, optional): Proportion of people to allocate to training set.
            Defaults to 0.7.
        val_ratio (float, optional): Proportion of people to allocate to validation set.
            Defaults to 0.15.
        test_ratio (float, optional): Proportion of people to allocate to test set.
            Defaults to 0.15.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        dict: A dictionary with keys 'train', 'valid', and 'test', where each value
            is a list of recording entries. Each recording entry has the same format
            as in the input dataset.
        
    Example:
        >>> dataset = scan_pd_dataset("Italian_Parkinsons_Voice_and_Speech")
        >>> splits = create_train_val_test_splits(dataset, 
        ...                                      train_ratio=0.8, 
        ...                                      val_ratio=0.1, 
        ...                                      test_ratio=0.1)
        >>> print(f"Training: {len(splits['train'])} files")
        >>> print(f"Validation: {len(splits['valid'])} files")
        >>> print(f"Test: {len(splits['test'])} files")
        Training: 280 files
        Validation: 35 files
        Test: 40 files
    """
    random.seed(seed)
    np.random.seed(seed)
    splits = {'train': [], 'valid': [], 'test': []}
    for grp, files in dataset.items():
        people = {}
        for f in files:
            person = f['key'].split('_')[0]
            people.setdefault(person, []).append(f)
        names = list(people)
        random.shuffle(names)
        n = len(names)
        t = int(n * train_ratio)
        v = int(n * val_ratio)
        for name in names[:t]:
            splits['train'].extend(people[name])
        for name in names[t:t + v]:
            splits['valid'].extend(people[name])
        for name in names[t + v:]:
            splits['test'].extend(people[name])
    for split, items in splits.items():
        print(f"  {split}: {len(items)} files")
    return splits


def create_manifest_files(
        splits,
        output_dir,
        rel_path=True,
        external_prefix="Italian_Parkinsons_Voice_and_Speech/italian_parkinson"):
    """Create JSON manifest files for train, validation, and test splits.
    
    This function creates JSON manifest files for each data split, with options
    to use relative or absolute paths. The manifest files contain metadata for
    each recording and are used by training pipelines.
    
    Args:
        splits (dict): The data splits as returned by create_train_val_test_splits().
        output_dir (str): Directory where manifest files will be saved.
        rel_path (bool, optional): Whether to use relative paths in the manifest.
            If True, paths will be prefixed with external_prefix. Defaults to True.
        external_prefix (str, optional): Prefix to add to relative paths.
            Defaults to "Italian_Parkinsons_Voice_and_Speech/italian_parkinson".
        
    Returns:
        None: This function doesn't return a value but creates manifest files
              on disk for each split (train.json, valid.json, test.json).
        
    Example:
        >>> dataset = scan_pd_dataset("data/Italian_PD")
        >>> splits = create_train_val_test_splits(dataset)
        >>> create_manifest_files(splits, "manifests")
        Created train manifest (280 entries) at manifests/train.json
        Created valid manifest (35 entries) at manifests/valid.json
        Created test manifest (40 entries) at manifests/test.json
    """
    for split, items in splits.items():
        manifest = {}
        for entry in items:
            info = entry['info']
            path = info['path']
            if rel_path:
                parts = path.split('/')
                if len(parts) > 2:
                    path = f"{external_prefix}/{'/'.join(parts[1:])}"
            manifest[entry['key']] = {
                'path': path,
                'group': info['group'],
                'sex': info['sex'],
                'age': int(info['age']),
                'duration': float(info['duration']),
                'label': str(info['label'])
            }
        out = os.path.join(output_dir, f"{split}.json")
        with open(out, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Created {split} manifest ({len(manifest)} entries) at {out}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--absolute_paths', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        parser.error("Ratios must sum to 1.0")

    ds = scan_pd_dataset(args.data_dir)
    splits = create_train_val_test_splits(
        ds,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed)
    create_manifest_files(
        splits,
        args.output_dir,
        rel_path=not args.absolute_paths)
    print("Done!")
