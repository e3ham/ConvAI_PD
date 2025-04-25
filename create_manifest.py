# #!/usr/bin/env python3
# import os
# import json
# import pandas as pd
# import torchaudio
# import numpy as np
# from tqdm import tqdm
# import argparse
# from pathlib import Path
# import random
# import glob


# def get_audio_duration(file_path):
#     """
#     Get the exact duration of an audio file in seconds using torchaudio
#     """
#     try:
#         info = torchaudio.info(file_path)
#         duration = info.num_frames / info.sample_rate
#         return duration  # Return exact duration with full precision
#     except Exception as e:
#         print(f"Error getting duration for {file_path}: {e}")
#         # Return default duration if there's an error
#         return 41.0


# def read_group_metadata(group_path):
#     """
#     Read metadata from Excel files at the group level and create a mapping
#     for person names/folders to their metadata
#     """
#     person_metadata = {}
    
#     # Find Excel files in the group directory
#     excel_files = glob.glob(os.path.join(group_path, "*.xlsx"))
    
#     for excel_file in excel_files:
#         file_name = os.path.basename(excel_file)
#         print(f"  Reading metadata from {file_name}")
#         print(excel_file, excel_files)
        
#         try:
#             # Read the Excel file into a pandas DataFrame
#             df = pd.read_excel(excel_file, engine="openpyxl", header=1)
            
#             # Skip if no data
#             if df.empty or 'name' not in df.columns:
#                 print(f"    No usable data found in {file_name}")
#                 continue
                
#             # Process main person data
#             for _, row in df.iterrows():
#                 if pd.isna(row['name']) or pd.isna(row['surname']):
#                     continue  # Skip rows with missing name data
                
#                 # Create keys for different possible folder naming patterns
#                 name = str(row['name']).strip()
#                 surname = str(row['surname']).strip()
                
#                 # Format 1: "First Last"
#                 key1 = f"{name} {surname}"
                
#                 # Format 2: "First L" (if surname is a single letter)
#                 key2 = f"{name} {surname[0]}" if len(surname) >= 1 else key1
                
#                 # Format 3: "First" (just in case)
#                 key3 = name
                
#                 # Store the metadata with all possible keys
#                 for key in [key1, key2, key3]:
#                     person_metadata[key] = {
#                         'name': name,
#                         'surname': surname,
#                         'sex': row.get('sex', 'M'),  # Default to 'M' if missing
#                         'age': int(row.get('age', 20)) if not pd.isna(row.get('age')) else 999,  # Default to 99 if missing
#                         'from': row.get('from', ''),
#                         'time1': row.get('time1', 0),
#                         'CPS1': row.get('CPS1', 0),
#                         'time2': row.get('time2', 0),
#                         'CPS2': row.get('CPS2', 0),
#                         'time3': row.get('time3', 0),
#                         'CPS3': row.get('CPS3', 0)
#                     }
                    
#             print(f"    Loaded metadata for {len(df)} people from {file_name}")
                
#         except Exception as e:
#             print(f"  Error reading metadata from {excel_file}: {e}")
    
#     return person_metadata


# def format_file_key(person_name, group_name, file_name):
#     """
#     Format file key as shown in the example:
#     "PERSON_NAME_GROUP_NAME_FILENAME.wav"
#     """
#     # Clean up person name (replace spaces with underscores)
#     person_name = person_name.replace(" ", "_").upper()
    
#     # Clean up group name (e.g., "15 Young Healthy Control" -> "YOUNG_HEALTHY_CONTROL")
#     group_parts = group_name.split()
#     if group_parts[0].isdigit():  # Remove the number prefix if present
#         group_parts = group_parts[1:]
#     group_name = "_".join(group_parts).upper()
    
#     # Extract the file type (PR1, B1, B2, etc.) and code from filename
#     file_parts = file_name.split(".")
#     file_code = file_parts[0]  # Remove extension
    
#     # Count the parts to determine if we need to add a counter
#     counter = ""
#     if "_" in file_code:
#         # If the file already has a counter, extract it
#         parts = file_code.split("_")
#         if len(parts) > 1 and parts[-1].isdigit():
#             counter = f"_{parts[-1]}"
#             file_code = "_".join(parts[:-1])
    
#     # Find or extract a number to use as counter if needed
#     if not counter:
#         # Try to extract a digit sequence from the filename
#         import re
#         matches = re.findall(r'(\d+)', file_code)
#         if matches:
#             # Use the last number sequence as counter
#             counter = f"_{matches[-1]}"
    
#     # Combine all parts
#     return f"{person_name}_{group_name}_{file_code}{counter}.wav"


# def scan_pd_dataset(data_dir):
#     """
#     Scan the PD dataset with its specific hierarchical structure
#     and incorporate metadata from Excel files at the group level
#     """
#     print(f"Scanning dataset at {data_dir}...")

#     # Initialize structure
#     dataset = {
#         "young_healthy": [],
#         "elderly_healthy": [],
#         "parkinsons": []
#     }

#     # Check if the main directory exists
#     if not os.path.isdir(data_dir):
#         raise ValueError(f"Data directory {data_dir} does not exist")

#     # Scan first level directories (health condition groups)
#     for group_dir in os.listdir(data_dir):
#         print("GROUP DIRECTORY SJFBHSJFDB KBSFJKSHB:", group_dir)
#         group_path = os.path.join(data_dir, group_dir)
#         print(group_path)

#         # Skip if not a directory
#         if not os.path.isdir(group_path):
#             continue

#         # Categorize by group name
#         if "Young Healthy" in group_dir:
#             group_key = "young_healthy"
#         elif "Elderly Healthy" in group_dir:
#             group_key = "elderly_healthy"
#         elif "Parkinson's" in group_dir:
#             group_key = "parkinsons"
#         else:
#             # Skip unknown directories
#             print(f"Skipping unknown directory: {group_dir}")
#             continue

#         print(f"Scanning {group_dir}...")
        
#         # Read group metadata from Excel files in the group directory
#         person_metadata = read_group_metadata(group_path)

#         # Scan second level (individual people)
#         for person_dir in os.listdir(group_path):
#             person_path = os.path.join(group_path, person_dir)

#             # Skip if not a directory or is an Excel file
#             if not os.path.isdir(person_path) or person_dir.endswith(".xlsx"):
#                 continue

#             print(f"  Scanning person: {person_dir}")
            
#             # Try to find this person's metadata
#             metadata = None
#             if person_dir in person_metadata:
#                 metadata = person_metadata[person_dir]
#                 print(f"    Found matching metadata for {person_dir}")
            
#             # Try other potential name formats if not found
#             if metadata is None:
#                 # Try partial matches
#                 for key in person_metadata:
#                     if person_dir in key or key in person_dir:
#                         metadata = person_metadata[key]
#                         print(f"    Found partial matching metadata for {person_dir} using {key}")
#                         break
            
#             # If still no metadata, create default
#             if metadata is None:
#                 print(f"    No metadata found for {person_dir}, using defaults")
#                 # Extract first name and last initial if possible
#                 parts = person_dir.split()
#                 if len(parts) >= 2:
#                     firstname = parts[0]
#                     lastname = parts[1]
#                 else:
#                     firstname = person_dir
#                     lastname = ""
                
#                 metadata = {
#                     'name': firstname,
#                     'surname': lastname,
#                     'sex': 'M',  # Default
#                     'age': 20,   # Default
#                     'from': '',
#                 }
            
#             # Scan for audio files
#             wav_files = [f for f in os.listdir(person_path) if f.endswith('.wav')]
            
#             for file_name in wav_files:
#                 file_path = os.path.join(person_path, file_name)
                
#                 # Get audio duration with full precision
#                 duration = get_audio_duration(file_path)
                
#                 # Create the formatted key for the manifest
#                 formatted_key = format_file_key(person_dir, group_dir, file_name)
                
#                 # Create file info
#                 file_info = {
#                     "path": file_path.replace(os.path.sep, "/"),  # Use forward slashes for consistency
#                     "group": group_dir,
#                     "sex": metadata.get("sex", "M"),
#                     "age": metadata.get("age", 20),
#                     "label": "PD" if "People" in group_dir else "HC",
#                     "duration": duration
#                 }
                
#                 # Add to dataset with person info
#                 dataset[group_key].append({
#                     "key": formatted_key,
#                     "info": file_info
#                 })

#     # Print statistics
#     print("\nDataset statistics:")
#     for group, files in dataset.items():
#         print(f"  {group}: {len(files)} files")

#     total_files = sum(len(files) for files in dataset.values())
#     print(f"  Total: {total_files} files")

#     return dataset


# def create_train_val_test_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
#     """
#     Create stratified train/validation/test splits, ensuring people are not split across sets
#     """
#     # Set random seed for reproducibility
#     random.seed(seed)
#     np.random.seed(seed)

#     # Initialize splits
#     splits = {
#         "train": [],
#         "valid": [],
#         "test": []
#     }

#     # Process each group separately to maintain stratification
#     for group, files in dataset.items():
#         # Get unique people by extracting from the formatted keys
#         people = {}
#         for file_data in files:
#             key = file_data["key"]
#             person = key.split("_")[0]  # First part of the key is the person
#             if person not in people:
#                 people[person] = []
#             people[person].append(file_data)
        
#         # Get the list of people
#         people_list = list(people.keys())
        
#         # Shuffle people
#         random.shuffle(people_list)

#         # Calculate splits
#         n_people = len(people_list)
#         n_train = int(n_people * train_ratio)
#         n_val = int(n_people * val_ratio)

#         # Split people
#         train_people = people_list[:n_train]
#         val_people = people_list[n_train:n_train + n_val]
#         test_people = people_list[n_train + n_val:]

#         # Assign files to splits based on person
#         for person in train_people:
#             splits["train"].extend(people[person])
        
#         for person in val_people:
#             splits["valid"].extend(people[person])
            
#         for person in test_people:
#             splits["test"].extend(people[person])

#     # Print split statistics
#     print("\nSplit statistics:")
#     for split_name, split_files in splits.items():
#         print(f"  {split_name}: {len(split_files)} files")

#     return splits


# # Modify this part in the scan_pd_dataset function
# # Replace the current file_info assignment with this:

# # And replace the create_manifest_files function with this:
# def create_manifest_files(splits, output_dir, rel_path=True):
#     """
#     Create the JSON manifest files with proper type handling to prevent boolean conversion
#     """
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Process each split
#     for split_name, split_files in splits.items():
#         # Initialize manifest
#         manifest = {}
        
#         # Add files to manifest
#         for file_data in split_files:
#             key = file_data["key"]
#             info = file_data["info"]
            
#             # Adjust path if needed
#             if rel_path:
#                 path = info["path"]
#                 # Create relative path with prefix
#                 common_prefix = "Italian_Parkinsons_Voice_and_Speech/italian_parkinson"
#                 parts = path.split("/")
#                 if len(parts) > 2:
#                     path = f"{common_prefix}/{'/'.join(parts[1:])}"
#             else:
#                 path = os.path.abspath(info["path"])
            
#             # IMPORTANT: Create a new dictionary to avoid reference issues
#             manifest_entry = {
#                 "path": path,
#                 "group": info["group"],
#                 "sex": info["sex"],
#                 "age": int(info["age"]),
#                 "duration": float(info["duration"])
#             }
            
#             # Handle label specially to prevent boolean conversion
#             label_str = str(info["label"])
#             # Force non-boolean JSON serialization by adding a prefix
#             # This ensures it won't be interpreted as boolean
#             manifest_entry["label"] = label_str
            
#             # Add to manifest
#             manifest[key] = manifest_entry
        
#         # Save manifest to JSON file
#         output_file = os.path.join(output_dir, f"{split_name}.json")
#         with open(output_file, 'w') as f:
#             json.dump(manifest, f, indent=2)
        
#         print(f"Created {split_name} manifest with {len(manifest)} entries at {output_file}")

#         # Debug: Check label types in the first few entries
#         print(f"\nChecking first 3 entries in {split_name} manifest:")
#         count = 0
#         for key, value in manifest.items():
#             if count < 3:
#                 print(f"  Entry '{key}': label='{value['label']}' (type: {type(value['label']).__name__})")
#                 count += 1


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Create JSON manifests for Parkinson's disease dataset")

#     # Add arguments
#     parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the PD dataset")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save JSON manifests")
#     parser.add_argument("--external_path", type=str, help="External path to use in manifests (default: Italian_Parkinsons_Voice_and_Speech)")
#     parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of data for training")
#     parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of data for validation")
#     parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of data for testing")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
#     parser.add_argument("--absolute_paths", action="store_true", help="Use absolute paths in manifests")

#     # Parse arguments
#     args = parser.parse_args()

#     # Validate ratios
#     if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
#         parser.error("Train, validation, and test ratios must sum to 1.0")

#     # Scan dataset
#     dataset = scan_pd_dataset(args.data_dir)

#     # Create splits
#     splits = create_train_val_test_splits(
#         dataset,
#         train_ratio=args.train_ratio,
#         val_ratio=args.val_ratio,
#         test_ratio=args.test_ratio,
#         seed=args.seed
#     )

#     # Create manifest files
#     create_manifest_files(splits, args.output_dir, rel_path=not args.absolute_paths)

#     print("Done!")

#!/usr/bin/env python3
import os
import json
import glob
import random
from pathlib import Path

import pandas as pd
import torchaudio
import numpy as np

def get_audio_duration(file_path):
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate

def read_group_metadata(group_dir_path):
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
            if 'time 1' in df.columns: df.rename(columns={'time 1': 'time1'}, inplace=True)
            if 'time 2' in df.columns: df.rename(columns={'time 2': 'time2'}, inplace=True)
            if 'time 3' in df.columns: df.rename(columns={'time 3': 'time3'}, inplace=True)
        except Exception as e:
            print(f"    Error loading {file_name}: {e}")
            continue

        if df.empty or 'name' not in df.columns:
            print(f"    No usable data found in {file_name}")
            continue
            
        # Handle the case where surname column doesn't exist
        if 'surname' not in df.columns:
            print(f"    'surname' column not found in {file_name}, creating empty column")
            df['surname'] = ""

        for _, row in df.iterrows():
            if pd.isna(row['name']):
                continue
                
            # Clean the name field
            name = str(row['name']).strip()
            
            # Clean the surname field - handle [object Object] and other oddities
            surname = str(row.get('surname', '')).strip()
            if surname.startswith('[object Object]') or pd.isna(row.get('surname')):
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
                age = int(row.get('age', 20)) if not pd.isna(row.get('age')) else 20
            except:
                age = 20
                
            # Create metadata entry
            metadata_value = {
                'sex':    row.get('sex', 'M'),
                'age':    age,
                'from':   row.get('from', ''),
                'time1':  row.get('time1', 0),
                'cps1':   row.get('cps1', 0) if 'cps1' in row else row.get('CPS1', 0),
                'time2':  row.get('time2', 0),
                'cps2':   row.get('cps2', 0) if 'cps2' in row else row.get('CPS2', 0),
                'time3':  row.get('time3', 0),
                'cps3':   row.get('cps3', 0) if 'cps3' in row else row.get('CPS3', 0),
            }
            
            # Store with multiple key formats to increase match chance
            person_metadata[lowercase_key] = metadata_value
            person_metadata[name.lower()] = metadata_value
            
            # Also store with name and first letter of surname (common format)
            if surname:
                person_metadata[f"{name.lower()} {surname[0].lower()}"] = metadata_value
                
            # Print what we're adding
            print(f"    Added metadata entry: '{uppercase_key}' -> {metadata_value['sex']}, {metadata_value['age']}")

        print(f"    Loaded metadata for {len(df)} rows from {file_name}")
    
    return person_metadata

def scan_pd_dataset(data_dir):
    print(f"Scanning dataset at {data_dir}...")
    dataset = {"young_healthy": [], "elderly_healthy": [], "parkinsons": []}

    for group_dir in os.listdir(data_dir):
        group_path = os.path.join(data_dir, group_dir)
        if not os.path.isdir(group_path): continue

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
        wav_files = glob.glob(os.path.join(group_path, '**', '*.wav'), recursive=True)
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
                last_initial = name_parts[1][0].lower() if len(name_parts[1]) > 0 else ""
                alt_key = f"{first_name} {last_initial}"
                if alt_key in metadata_map:
                    meta = metadata_map[alt_key]
            
            # If still no match, use default metadata and report the miss only once
            if meta is None:
                if person_dir not in missing_metadata:
                    missing_metadata.add(person_dir)
                    print(f"    No metadata match for '{person_dir}', tried: '{lookup_key}'")
                    if len(name_parts) >= 2:
                        print(f"      Also tried: '{first_name}' and '{alt_key}'")
                
                # Parse person_dir for default metadata
                parts = person_dir.split(maxsplit=1)
                firstname = parts[0]
                lastname = parts[1] if len(parts) > 1 else ''
                meta = {'sex': 'M', 'age': 20, 'from': ''}
            
            duration = get_audio_duration(file_path)
            key = format_file_key(person_dir, group_dir, os.path.basename(file_path))
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
            print(f"  Missing metadata for {len(missing_metadata)} people in this group.")

    for k, v in dataset.items():
        print(f"  {k}: {len(v)} files")
    return dataset


def create_train_val_test_splits(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed); np.random.seed(seed)
    splits = {'train':[], 'valid':[], 'test':[]}
    for grp, files in dataset.items():
        people = {}
        for f in files:
            person = f['key'].split('_')[0]
            people.setdefault(person, []).append(f)
        names = list(people);
        random.shuffle(names)
        n = len(names)
        t = int(n*train_ratio); v = int(n*val_ratio)
        for name in names[:t]: splits['train'].extend(people[name])
        for name in names[t:t+v]: splits['valid'].extend(people[name])
        for name in names[t+v:]: splits['test'].extend(people[name])
    for split, items in splits.items(): print(f"  {split}: {len(items)} files")
    return splits


def create_manifest_files(splits, output_dir, rel_path=True, external_prefix="Italian_Parkinsons_Voice_and_Speech/italian_parkinson"):
    os.makedirs(output_dir, exist_ok=True)
    for split, items in splits.items():
        manifest = {}
        for entry in items:
            info = entry['info']
            path = info['path']
            if rel_path:
                parts = path.split('/')
                if len(parts)>2:
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
        with open(out, 'w') as f: json.dump(manifest, f, indent=2)
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
    splits = create_train_val_test_splits(ds, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    create_manifest_files(splits, args.output_dir, rel_path=not args.absolute_paths)
    print("Done!")
    