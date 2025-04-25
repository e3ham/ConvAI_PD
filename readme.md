# Italian Parkinson's Voice Dataset Processing

This README documents the processing pipeline for the Italian Parkinson's Voice and Speech dataset for Parkinson's Disease (PD) detection using voice recordings.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Directory Structure](#directory-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Data Augmentation](#data-augmentation)
5. [Task Separation](#task-separation)
6. [Manifest Creation](#manifest-creation)
7. [Training Models](#training-models)
8. [Troubleshooting](#troubleshooting)

## Dataset Overview

The Italian Parkinson's Voice and Speech dataset contains recordings of people with Parkinson's Disease (PD) and Healthy Controls (HC). The dataset includes:

- 15 Young Healthy Controls
- 22 Elderly Healthy Controls
- 28 People with Parkinson's Disease

Each participant performed several speaking tasks, including:
- Reading tasks (files starting with B1 or B2)
- Other speech tasks (files with other prefixes like PR1, VA1, etc.)

To obtain the dataset (and all variations), please visit https://drive.google.com/drive/folders/1XMhdukiEIeZlqvFq95QGWpeH5mQykZJc?usp=sharing. 

## Directory Structure

```
project_root/
├── pd_dataset/                        # Original dataset
│   ├── 15 Young Healthy Control/
│   │   ├── Speaker1/
│   │   │   ├── B1xxx.wav
│   │   │   ├── B2xxx.wav
│   │   │   └── ...
│   ├── 22 Elderly Healthy Control/
│   └── 28 People with Parkinson's disease/
├── augmented_data/                    # Augmented audio files
├── manifests/                         # Original manifests
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── aug_manifests/                     # Manifests with augmented files
├── task_separated/                    # Task-specific datasets
│   ├── reading_task/
│   ├── other_task/
│   ├── reading_manifests/
│   └── other_manifests/
└── results/                           # Training results
```

## Data Preprocessing

### Creating Initial Manifests

Create manifests from the original dataset:

```bash
python create_manifest.py --data_dir "pd_dataset" --output_dir "manifests"
```

This script:
- Scans the dataset directory structure
- Extracts metadata from Excel files
- Creates train/validation/test splits
- Saves manifests as JSON files

### Handling Excel Metadata

The script reads metadata from Excel files in each group directory:
- Handles different formats of Excel files
- Normalizes metadata (names, ages, sex)
- Maps metadata to speaker folders
- Fixes capitalization mismatches using case-insensitive comparison

```python
# Example of case-insensitive metadata matching
meta = metadata_map.get(person_dir.strip().lower())
```

## Data Augmentation

Augment voice recordings to improve model performance:

```bash
python data_augment.py --data_dir pd_dataset --output_dir augmented_data --manifest_dir manifests
```

### Augmentation Techniques

The augmentation script applies:

1. **General Audio Augmentations**:
   - Time stretching (0.9-1.1x)
   - Pitch shifting (±2 semitones)
   - Adding noise (low-level)
   - High-pass/low-pass filtering

2. **Parkinson's-Specific Augmentations**:
   - Vocal tremor (4-7 Hz modulation)
   - Breathiness (simulates reduced vocal fold closure)
   - Reduced articulation (spectral modifications)
   - Jitter/shimmer (micro-perturbations in frequency/amplitude)

### Directory Structure Preservation

The augmentation preserves the original directory structure:

```
augmented_data/
├── 15 Young Healthy Control/
│   ├── Speaker1/
│   │   ├── original_file_aug1.wav
│   │   ├── original_file_aug2.wav
```

## Task Separation

Separate files into reading tasks (B1/B2) and other tasks:

```bash
python task-separation-script.py \
  --data_dir pd_dataset \
  --output_dir task_separated \
  --manifests_dir manifests \
  --augmented_dir augmented_data
```

This creates:
- `reading_task/` - Contains all B1/B2 files
- `other_task/` - Contains all other files
- Task-specific manifests for training

## Manifest Creation

### Original Manifests

The original manifests (`train.json`, `valid.json`, `test.json`) have entries like:

```json
"GIUSEPPE_ANDREA_M_YOUNG_HEALTHY_CONTROL_PR1LBULCAAS94M100120171021_100120171021.wav": {
  "path": "pd_dataset/15 Young Healthy Control/Giuseppe Andrea M/PR1LBULCAAS94M100120171021.wav",
  "group": "15 Young Healthy Control",
  "sex": "M",
  "age": 21,
  "duration": 40.958625,
  "label": "HC"
}
```

### Augmented Manifests

Create manifests that include augmented files:

```bash
python create-augmented-manifest.py \
  --manifest_dir manifests \
  --augmented_dir augmented_data \
  --output_dir aug_manifests
```

This generates manifests that include both original and augmented files.

### Task-Specific Manifests

The task separation script creates separate manifests for each task:
- `reading_manifests/train.json` - Reading task (B1/B2)
- `other_manifests/train.json` - Other speech tasks

## Training Models

### Configuration Changes

For reading tasks:

```yaml
train_annotation: "task_separated/reading_manifests/train.json"
valid_annotation: "task_separated/reading_manifests/valid.json"
test_annotation: "task_separated/reading_manifests/test.json"
data_root: "task_separated/reading_task"
output_folder: results/ecapa_tdnn_pd_detection_reading/<seed>
```

For other tasks:

```yaml
train_annotation: "task_separated/other_manifests/train.json"
valid_annotation: "task_separated/other_manifests/valid.json"
test_annotation: "task_separated/other_manifests/test.json"
data_root: "task_separated/other_task"
output_folder: results/ecapa_tdnn_pd_detection_other/<seed>
```

### Fixed Audio Pipeline

The training script uses an improved audio pipeline that handles missing files:

```python
@sb.utils.data_pipeline.takes("path")
@sb.utils.data_pipeline.provides("sig")
def audio_pipeline(path):
    """Handles missing files by finding alternatives."""
    # Check if file exists
    if not os.path.exists(path) and "_aug" in path:
        # Fall back to original file if augmented doesn't exist
        original_path = path.replace("_aug1.wav", ".wav").replace("_aug2.wav", ".wav")
        if os.path.exists(original_path):
            print(f"Using original instead: {os.path.basename(path)} -> {os.path.basename(original_path)}")
            path = original_path
    
    # Load audio
    try:
        sig, fs = torchaudio.load(path)
        sig = torchaudio.functional.resample(sig, orig_freq=fs, new_freq=16000).squeeze(0)
        return sig
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return torch.zeros(16000)  # Return dummy signal
```

Running Models
We provide implementations for two experimental setups: a baseline multimodal model combining Whisper embeddings with acoustic features, and a version using data augmentation.
Environment Setup
First, ensure you have the required dependencies:
bashpip install speechbrain torchaudio matplotlib numpy pandas scikit-learn praat-parselmouth
Experiment 1: Baseline Multimodal Model
The baseline model combines Whisper embeddings with acoustic features (pitch, jitter, shimmer):
bash# Navigate to the implementation directory
cd whisper_acoustic_features

# Train the baseline model
python whisper.py whisper.yaml

# To run with a specific seed
python whisper.py whisper.yaml --seed 1986
Class-Weighted Variant
To address class imbalance, we can run the model with class weighting:
bash# Edit whisper.yaml to include class weights:
# compute_cost: !new:torch.nn.CrossEntropyLoss
#   weight: !new:torch.Tensor [[0.3, 0.7]]

# Then run:
python whisper.py whisper.yaml
Experiment 2: Training with Augmented Data
To train with the augmented dataset:
bash# Use augmented manifests
python whisper.py whisper.yaml \
  --train_annotation aug_manifests/train.json \
  --valid_annotation aug_manifests/valid.json \
  --test_annotation aug_manifests/test.json

# To start fresh (removing existing checkpoints)
python whisper.py whisper.yaml --fresh_start
Customizing Training
Key parameters in whisper.yaml you may want to modify:
yaml# Model size
whisper_source: "openai/whisper-base"  # Options: tiny, base, small, medium

# Training parameters
number_of_epochs: 10
batch_size: 8
lr: 0.0001
lr_ssl: 0.00001

# Whether to freeze the Whisper encoder
freeze_ssl: True  # Set to False to fine-tune
Task-Specific Models
For training on specific tasks:
bash# Reading task model
python whisper.py whisper.yaml \
  --train_annotation task_separated/reading_manifests/train.json \
  --valid_annotation task_separated/reading_manifests/valid.json \
  --test_annotation task_separated/reading_manifests/test.json \
  --data_folder task_separated/reading_task \
  --output_folder results/models/whisper_reading

# Other tasks model
python whisper.py whisper.yaml \
  --train_annotation task_separated/other_manifests/train.json \
  --valid_annotation task_separated/other_manifests/valid.json \
  --test_annotation task_separated/other_manifests/test.json \
  --data_folder task_separated/other_task \
  --output_folder results/models/whisper_other
Results Interpretation
After training, results are saved in the output folder (default: results/models/whisper/1986/):
results/models/whisper/1986/
├── save/               # Model checkpoints
├── plots/              # Performance visualizations
│   ├── loss_curves.png              # Training/validation curves
│   ├── confusion_matrix.png         # Test set confusion matrix
│   ├── demographic_epoch_*.png      # Performance by demographic
│   ├── sex_performance.png          # Male vs. female accuracy
│   └── dataset_performance.png      # HC vs. PD accuracy trends 
├── train_log.txt       # Training progress log
├── train_losses.json   # Training loss values
└── test_error.json     # Final test error rate
Key metrics to examine:

Overall test accuracy (test_error.json)
Class-specific performance (HC vs. PD accuracy)
Demographic performance patterns (age, gender)
Training/validation loss curves (for convergence and overfitting)

Our baseline model achieved 62.21% overall accuracy with 93.33% HC and 45.54% PD accuracy. The class-weighted variant improved to 68.60% overall with more balanced 61.67% HC and 72.32% PD accuracy.

## Troubleshooting

### Missing Files

If you encounter "File does not exist" warnings:

1. **Check relative vs. absolute paths** - Ensure your manifests use the correct path format
2. **Update manifests** - Use the comprehensive manifest creator script
3. **Fix audio pipeline** - Implement the robust audio pipeline with fallback mechanisms

### Tensor Size Mismatches

For augmentation errors like "tensor a (670225) must match tensor b (670224)":

1. **Fix breathiness function** - Ensure exact tensor length matches
2. **Use the ensure_length_match helper** - Makes sure processed arrays match original lengths
3. **Add proper error handling** - Catch exceptions in augmentation functions

### Other Issues

- **Excel metadata issues** - Use case-insensitive matching for speaker names
- **Path construction** - Use Path objects for more robust path handling
- **Task separation** - Use symlinks instead of copies to save disk space

## References

- SpeechBrain: [https://speechbrain.github.io/](https://speechbrain.github.io/)
- Librosa: [https://librosa.org/](https://librosa.org/)
- Torchaudio: [https://pytorch.org/audio/](https://pytorch.org/audio/)