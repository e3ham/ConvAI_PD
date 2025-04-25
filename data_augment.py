#!/usr/bin/env python3
import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import random
import librosa
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm  # Add tqdm for progress bars

# AudioAugmenter and PDVoiceAugmenter classes remain the same
# ... (include all the code for these classes from the previous version)

class AudioAugmenter:
    """Class to handle audio augmentation techniques for Parkinson's disease detection."""
    
    def __init__(self, sample_rate=16000, augmentation_factor=2):
        """
        Initialize the augmenter.
        
        Args:
            sample_rate: Target sample rate for audio processing
            augmentation_factor: How many augmented versions to create per original file
        """
        self.sample_rate = sample_rate
        self.augmentation_factor = augmentation_factor
        
    def load_audio(self, file_path):
        """Load audio file and convert to target sample rate."""
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def save_audio(self, waveform, file_path):
        """Save the augmented waveform to disk."""
        torchaudio.save(file_path, waveform, self.sample_rate)
        
    def time_stretch(self, waveform, rate_range=(0.9, 1.1)):
        """Apply time stretching without changing pitch."""
        rate = random.uniform(*rate_range)
        # Convert to numpy for processing with librosa
        audio_np = waveform.numpy().squeeze()
        # Time stretch
        stretched = librosa.effects.time_stretch(audio_np, rate=rate)
        # Convert back to tensor - make a copy to ensure positive strides
        return torch.from_numpy(stretched.copy()).unsqueeze(0)
    
    def pitch_shift(self, waveform, semitone_range=(-2, 2)):
        """Apply pitch shifting without changing duration."""
        n_steps = random.uniform(*semitone_range)
        # Convert to numpy for processing
        audio_np = waveform.numpy().squeeze()
        # Pitch shift
        shifted = librosa.effects.pitch_shift(
            audio_np, 
            sr=self.sample_rate, 
            n_steps=n_steps
        )
        # Convert back to tensor - make a copy to ensure positive strides
        return torch.from_numpy(shifted.copy()).unsqueeze(0)
    
    def add_noise(self, waveform, noise_level_range=(0.001, 0.005)):
        """Add random noise to the waveform."""
        noise_level = random.uniform(*noise_level_range)
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def apply_filter(self, waveform, filter_type='highpass', cutoff_freq=100):
        """Apply a filter to the waveform."""
        audio_np = waveform.numpy().squeeze()
        
        # Butter filter design
        nyquist = 0.5 * self.sample_rate
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normalized_cutoff, btype=filter_type)
        
        # Apply filter
        filtered = filtfilt(b, a, audio_np)
        
        # Make a copy to ensure positive strides before converting to tensor
        filtered = filtered.copy()
        
        return torch.from_numpy(filtered).unsqueeze(0)
    
    def apply_random_augmentation(self, waveform):
        """Apply a randomly selected augmentation technique."""
        augmentation_techniques = [
            self.time_stretch,
            self.pitch_shift,
            self.add_noise,
            lambda w: self.apply_filter(w, 'highpass', random.uniform(80, 120)),
            lambda w: self.apply_filter(w, 'lowpass', random.uniform(7000, 8000))
        ]
        
        # Choose 1-2 random augmentations to apply
        n_augmentations = random.randint(1, 2)
        selected_augmentations = random.sample(augmentation_techniques, n_augmentations)
        
        # Apply the selected augmentations
        augmented_waveform = waveform
        for augmentation in selected_augmentations:
            augmented_waveform = augmentation(augmented_waveform)
            
        return augmented_waveform


class PDVoiceAugmenter:
    """
    Specialized augmentation techniques that simulate voice characteristics 
    associated with Parkinson's disease.
    """
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def ensure_length_match(self, processed_array, original_shape):
        """Helper function to ensure processed arrays match original length."""
        target_length = original_shape[0]
        
        if len(processed_array) > target_length:
            return processed_array[:target_length]
        elif len(processed_array) < target_length:
            padding = np.zeros(target_length - len(processed_array))
            return np.concatenate([processed_array, padding])
        return processed_array
    
    def add_tremor(self, waveform, rate_range=(4, 7), depth_range=(0.005, 0.02)):
        """
        Add vocal tremor - amplitude modulation at 4-7 Hz, 
        a common symptom in Parkinson's disease.
        """
        # Convert to numpy for processing
        audio_np = waveform.numpy().squeeze()
        original_shape = audio_np.shape
        
        # Generate tremor parameters
        tremor_rate = random.uniform(*rate_range)  # Hz
        tremor_depth = random.uniform(*depth_range)
        
        # Generate tremor modulation signal
        t = np.arange(0, len(audio_np)) / self.sample_rate
        tremor = 1.0 + tremor_depth * np.sin(2 * np.pi * tremor_rate * t)
        
        # Ensure tremor matches audio length
        if len(tremor) != len(audio_np):
            tremor = self.ensure_length_match(tremor, original_shape)
        
        # Apply tremor modulation
        modulated = audio_np * tremor
        
        # Convert back to tensor - make a copy to ensure positive strides
        return torch.from_numpy(modulated.copy()).unsqueeze(0)
    
    def add_breathiness(self, waveform, noise_level_range=(0.01, 0.05)):
        """
        Add breathiness to voice - common in Parkinson's due to reduced 
        vocal fold closure and air escape during phonation.
        """
        noise_level = random.uniform(*noise_level_range)
        
        # Get exact waveform length to ensure matching dimensions
        exact_length = waveform.shape[1]
        
        # Generate colored noise with exact length
        noise = np.random.randn(exact_length)
        
        # Apply 1/f filter to create pink noise (better matches breathiness)
        noise_fft = np.fft.rfft(noise)
        f = np.fft.rfftfreq(len(noise))
        f[0] = 1  # Avoid division by zero
        pink_filter = 1 / np.sqrt(f)
        noise_fft = noise_fft * pink_filter
        pink_noise = np.fft.irfft(noise_fft)
        
        # Ensure exact length match
        if len(pink_noise) != exact_length:
            pink_noise = pink_noise[:exact_length] if len(pink_noise) > exact_length else np.pad(
                pink_noise, (0, exact_length - len(pink_noise)), 'constant'
            )
        
        # Normalize and scale
        max_val = np.max(np.abs(pink_noise))
        if max_val > 0:  # Avoid division by zero
            pink_noise = pink_noise / max_val * noise_level
        
        # Convert to tensor and add to signal
        noise_tensor = torch.from_numpy(pink_noise.copy()).float().unsqueeze(0)
        
        return waveform + noise_tensor
    
    def reduce_articulation(self, waveform, factor_range=(0.7, 0.9)):
        """
        Simulate reduced articulation by applying mild low-pass filtering,
        which mimics the reduced precision in consonant production.
        """
        factor = random.uniform(*factor_range)
        
        # Convert to numpy for processing
        audio_np = waveform.numpy().squeeze()
        original_shape = audio_np.shape
        
        try:
            # Apply spectral envelope manipulation
            S = librosa.stft(audio_np)
            S_db = librosa.amplitude_to_db(np.abs(S))
            
            # Create weighting that reduces high frequencies more
            freq_bins = S_db.shape[0]
            weights = np.linspace(1, factor, freq_bins)
            weights = weights.reshape(-1, 1)  # Column vector
            
            # Apply weights
            S_db_weighted = S_db * weights
            
            # Convert back to time domain
            S_weighted = librosa.db_to_amplitude(S_db_weighted) * np.exp(1j * np.angle(S))
            reduced = librosa.istft(S_weighted)
            
            # Ensure the result matches the original length
            reduced = self.ensure_length_match(reduced, original_shape)
            
            # Convert back to tensor - make a copy to ensure positive strides
            return torch.from_numpy(reduced.copy()).unsqueeze(0)
        
        except Exception as e:
            print(f"Warning in reduce_articulation: {e}")
            return waveform
    
    def jitter_shimmer(self, waveform, jitter_range=(0.005, 0.02), shimmer_range=(0.04, 0.1)):
        """
        Add jitter (pitch perturbation) and shimmer (amplitude perturbation),
        which are increased in Parkinson's disease voices.
        """
        # This is a more complex operation that can fail in various ways
        # We'll handle it with more care
        
        try:
            jitter_amount = random.uniform(*jitter_range)
            shimmer_amount = random.uniform(*shimmer_range)
            
            # Convert to numpy for processing
            audio_np = waveform.numpy().squeeze()
            original_shape = audio_np.shape
            
            # Use a simpler approach - create local pitch/amplitude variations
            # Apply subtle jitter (small random variations to the signal)
            jitter = np.random.normal(0, jitter_amount, size=len(audio_np))
            jittered = audio_np + jitter * np.abs(audio_np)
            
            # Apply shimmer (amplitude variation)
            shimmer_env = 1 + shimmer_amount * np.sin(2 * np.pi * np.linspace(0, 10, len(audio_np)))
            shimmer_result = jittered * shimmer_env
            
            # Ensure final result matches original length
            shimmer_result = self.ensure_length_match(shimmer_result, original_shape)
            
            # Return the processed audio
            return torch.from_numpy(shimmer_result.copy()).unsqueeze(0)
            
        except Exception as e:
            print(f"Warning in jitter_shimmer: {e}")
            return waveform
    
    def apply_pd_augmentation(self, waveform):
        """Apply random Parkinson's-related augmentations to the waveform."""
        augmentation_techniques = [
            self.add_tremor,
            self.add_breathiness,
            self.reduce_articulation,
            self.jitter_shimmer
        ]
        
        # Choose 1-2 random PD-related augmentations to apply
        n_augmentations = random.randint(1, 2)
        selected_augmentations = random.sample(augmentation_techniques, n_augmentations)
        
        # Apply the selected augmentations
        augmented_waveform = waveform
        for augmentation in selected_augmentations:
            try:
                augmented_waveform = augmentation(augmented_waveform)
            except Exception as e:
                print(f"Warning: PD augmentation {augmentation.__name__} failed: {e}. Skipping.")
            
        return augmented_waveform


def find_actual_data_root(data_dir, file_path):
    """
    Find the actual root directory containing the data based on the file path.
    This helps with directory structure preservation when file paths in the manifest
    don't directly start with data_dir.
    """
    # Convert to absolute paths for comparison
    abs_data_dir = os.path.abspath(data_dir)
    abs_file_path = os.path.abspath(file_path) if os.path.exists(file_path) else file_path
    
    # If the file path already starts with data_dir, we're good
    if abs_file_path.startswith(abs_data_dir):
        return abs_data_dir
    
    # Try to find common parent directories
    parts_data = Path(abs_data_dir).parts
    parts_file = Path(abs_file_path).parts
    
    # Find the longest common prefix
    common_prefix = []
    for d, f in zip(parts_data, parts_file):
        if d == f:
            common_prefix.append(d)
        else:
            break
    
    if common_prefix:
        # Return the common parent directory
        return os.path.join(*common_prefix)
    
    # If no common parent, use just the data_dir
    return abs_data_dir


def get_relative_path(file_path, base_dir):
    """Get the relative path of a file from a base directory, preserving directory structure."""
    # Convert to absolute paths
    abs_file = os.path.abspath(file_path) if os.path.exists(file_path) else file_path
    abs_base = os.path.abspath(base_dir)
    
    # If the file is directly under the base, get the relative path
    if abs_file.startswith(abs_base):
        return os.path.relpath(abs_file, abs_base)
    
    # If not directly under base, try to find common structure
    parts_base = Path(abs_base).parts
    parts_file = Path(abs_file).parts
    
    # Find where the paths start to differ
    i = 0
    for i, (b, f) in enumerate(zip(parts_base, parts_file)):
        if b != f:
            break
    
    # If we found a common prefix, return the relative part
    if i > 0:
        # Get the non-common part of the file path
        rel_parts = parts_file[i:]
        return os.path.join(*rel_parts)
    
    # If no common structure, just return the filename
    return os.path.basename(file_path)


def augment_dataset(data_dir, output_dir, metadata_dir=None, augmentation_factor=2):
    """
    Augment the entire dataset and create new manifests, preserving original directory structure.
    
    Args:
        data_dir: Directory containing the original dataset
        output_dir: Directory to save augmented files
        metadata_dir: Directory containing the original metadata/manifests
        augmentation_factor: How many augmented versions to create per original file
    """
    augmenter = AudioAugmenter(augmentation_factor=augmentation_factor)
    pd_augmenter = PDVoiceAugmenter()
    
    # Handle existing manifests if available
    original_manifests = {}
    if metadata_dir and os.path.exists(metadata_dir):
        manifest_files = []
        for split in ['train', 'valid', 'test']:
            manifest_path = os.path.join(metadata_dir, f"{split}.json")
            if os.path.exists(manifest_path):
                manifest_files.append((split, manifest_path))
        
        # Load manifests with progress bar
        for split, path in tqdm(manifest_files, desc="Loading manifests"):
            with open(path, 'r') as f:
                original_manifests[split] = json.load(f)
            print(f"Loaded {split} manifest with {len(original_manifests[split])} entries")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    aug_manifests_dir = os.path.join(output_dir, 'manifests')
    os.makedirs(aug_manifests_dir, exist_ok=True)
    
    # Only augment the training data (common practice)
    if 'train' in original_manifests:
        augmented_manifest = {}
        
        # Include all original entries
        augmented_manifest.update(original_manifests['train'])
        
        # Track statistics
        pd_count = 0
        hc_count = 0
        
        # Get all files to process
        train_files = list(original_manifests['train'].items())
        
        # Process each file in the training set with a progress bar
        with tqdm(total=len(train_files), desc="Augmenting training files", unit="file") as pbar:
            for key, info in train_files:
                file_path = info['path']
                
                # Check if path exists, try with data_dir prefix if not
                actual_file_path = file_path
                if not os.path.exists(file_path):
                    # Try with data_dir prefix
                    potential_path = os.path.join(data_dir, file_path)
                    if os.path.exists(potential_path):
                        actual_file_path = potential_path
                    else:
                        print(f"Warning: Could not find {file_path} or {potential_path}, skipping")
                        pbar.update(1)
                        continue
                
                # Determine the actual data root to preserve exact directory structure
                actual_data_root = find_actual_data_root(data_dir, actual_file_path)
                
                # Get the path relative to the data root - this preserves directory structure
                rel_path = get_relative_path(actual_file_path, actual_data_root)
                
                # Create output directory that exactly mirrors the original structure
                output_subdir = os.path.join(output_dir, os.path.dirname(rel_path))
                os.makedirs(output_subdir, exist_ok=True)
                
                # Augment the file
                try:
                    # Load audio
                    waveform = augmenter.load_audio(actual_file_path)
                    
                    # Determine if this is a PD sample
                    is_pd = info['label'] == 'PD'
                    
                    # Create augmented versions with progress bar for multiple versions
                    for i in range(augmentation_factor):
                        # Update progress bar description to show current file
                        pbar.set_description(f"File: {os.path.basename(actual_file_path)} ({i+1}/{augmentation_factor})")
                        
                        # First apply general audio augmentations
                        aug_waveform = augmenter.apply_random_augmentation(waveform)
                        
                        # Apply PD-specific augmentations with 50% probability
                        # Always for PD samples, 50% for healthy controls
                        if is_pd or random.random() < 0.5:
                            try:
                                aug_waveform = pd_augmenter.apply_pd_augmentation(aug_waveform)
                            except Exception as e:
                                print(f"PD augmentation failed: {e}, using standard augmentation only")
                        
                        # Save the augmented audio with original directory structure
                        base_name = os.path.basename(actual_file_path)
                        name, ext = os.path.splitext(base_name)
                        aug_path = os.path.join(output_subdir, f"{name}_aug{i+1}{ext}")
                        
                        augmenter.save_audio(aug_waveform, aug_path)
                        
                        # Create output path for manifest that matches original pattern
                        # This ensures consistency between original and augmented paths
                        manifest_path = aug_path
                        if os.path.isabs(manifest_path) and os.path.isabs(file_path):
                            # If both are absolute, maintain the same base
                            manifest_path = os.path.join(
                                os.path.dirname(file_path), 
                                f"{name}_aug{i+1}{ext}"
                            )
                        elif not os.path.isabs(file_path):
                            # If original path is relative, keep output relative
                            manifest_path = os.path.join(
                                os.path.dirname(rel_path),
                                f"{name}_aug{i+1}{ext}"
                            )
                        
                        # Add to manifest
                        aug_key = f"{key.split('.')[0]}_aug{i+1}.wav"
                        augmented_manifest[aug_key] = {
                            'path': manifest_path.replace(os.sep, '/'),
                            'group': info['group'],
                            'sex': info['sex'],
                            'age': info['age'],
                            'label': info['label'],
                            'duration': info['duration'],
                            'augmented': True  # Flag to identify augmented samples
                        }
                        
                        # Track counts by label
                        if is_pd:
                            pd_count += 1
                        else:
                            hc_count += 1
                        
                except Exception as e:
                    print(f"Error augmenting {file_path}: {e}")
                
                pbar.update(1)
        
        # Save the augmented train manifest
        print("Saving augmented manifest...")
        with open(os.path.join(aug_manifests_dir, 'train_augmented.json'), 'w') as f:
            json.dump(augmented_manifest, f, indent=2)
            
        print(f"Augmented training set: {len(original_manifests['train'])} original files + "
              f"{pd_count + hc_count} augmented files = {len(augmented_manifest)} total files")
        print(f"  - PD samples: {pd_count} augmented")
        print(f"  - HC samples: {hc_count} augmented")
    
    # Copy validation and test manifests as-is (no augmentation)
    for split in tqdm(['valid', 'test'], desc="Copying evaluation manifests"):
        if split in original_manifests:
            with open(os.path.join(aug_manifests_dir, f"{split}.json"), 'w') as f:
                json.dump(original_manifests[split], f, indent=2)
            print(f"Copied {split} manifest: {len(original_manifests[split])} files")
    
    print("Dataset augmentation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment audio dataset for Parkinson's disease detection")
    parser.add_argument('--data_dir', required=True, help='Directory containing the original dataset')
    parser.add_argument('--output_dir', required=True, help='Directory to save augmented files')
    parser.add_argument('--manifest_dir', required=True, help='Directory containing the original manifests')
    parser.add_argument('--augmentation_factor', type=int, default=2, 
                        help='Number of augmented versions per original file')
    parser.add_argument('--no_progress_bar', action='store_true',
                        help='Disable progress bars (useful for logging to file)')
    
    args = parser.parse_args()
    
    # Disable progress bars if requested
    if args.no_progress_bar:
        from functools import partial
        tqdm = partial(lambda x, **kwargs: x, disable=True)
    
    augment_dataset(
        args.data_dir,
        args.output_dir,
        args.manifest_dir,
        args.augmentation_factor
    )