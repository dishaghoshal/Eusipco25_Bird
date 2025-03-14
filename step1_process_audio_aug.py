import os
import glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from collections import defaultdict
import audiomentations
from tqdm import tqdm
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Path to the directory containing the MP3 files
data_dir = "/export/home/anandr/bird_proj/mp3_data/"

# Define the augmentation pipeline
augmenter = audiomentations.Compose([
    audiomentations.AddGaussianNoise(min_amplitude=0.0005, max_amplitude=0.008, p=0.5),
    audiomentations.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.5),
    audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    audiomentations.Shift(min_shift=-0.15, max_shift=0.15, p=0.5),
    audiomentations.Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
    audiomentations.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=0.3),
    audiomentations.HighPassFilter(min_cutoff_freq=200, max_cutoff_freq=1000, p=0.3),
    audiomentations.Normalize(p=1.0),  # Always normalize for consistency
])

# Number of augmentations to generate per 001 file
NUM_AUGMENTATIONS = 50

# Create output directories
output_base_dir = "/export/home/anandr/bird_proj/processed_audios/"
train_dir = os.path.join(output_base_dir, "train")
test_dir = os.path.join(output_base_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def get_species_from_filename(filename):
    """Extract species name from the filename"""
    base_name = os.path.basename(filename)
    # Extract everything before the underscore
    return base_name.rsplit('_', 1)[0]

def is_001_file(filename):
    """Check if the file ends with _001.mp3"""
    return filename.endswith('_001.mp3')

def apply_augmentations(audio, sr, num_augmentations):
    """Apply audio augmentations and return a list of augmented samples"""
    augmented_samples = []
    
    for i in range(num_augmentations):
        # Apply augmentation
        augmented_audio = augmenter(samples=audio, sample_rate=sr)
        augmented_samples.append(augmented_audio)
    
    return augmented_samples

def process_audio_files():
    """Process audio files - group by species, filter, and split into train/test"""
    print("Searching for MP3 files...")
    all_mp3_files = glob.glob(os.path.join(data_dir, "*.mp3"))
    print(f"Found {len(all_mp3_files)} MP3 files.")
    
    # Group files by species
    species_files = defaultdict(list)
    for file_path in all_mp3_files:
        species = get_species_from_filename(file_path)
        species_files[species].append(file_path)
    
    # Filter species with at least 2 samples
    valid_species = {species: files for species, files in species_files.items() if len(files) >= 2}
    print(f"Found {len(valid_species)} species with at least 2 samples.")
    
    # Process files for each valid species
    train_files = []
    test_files = []
    
    for species, files in tqdm(valid_species.items(), desc="Processing species"):
        # Split files into 001 files (train with augmentation) and others (test)
        train_candidates = [f for f in files if is_001_file(f)]
        test_candidates = [f for f in files if not is_001_file(f)]
        
        if not train_candidates:
            print(f"Warning: No _001 file found for species {species}. Skipping.")
            continue
        
        # Process training files (001 files with augmentation)
        for train_file in train_candidates:
            # Load the audio file
            try:
                audio, sr = librosa.load(train_file, sr=None)
                
                # Create output filename
                base_name = os.path.basename(train_file)
                train_output_path = os.path.join(train_dir, base_name)
                
                # Save original file to train directory
                sf.write(train_output_path, audio, sr)
                train_files.append(train_output_path)
                
                # Generate and save augmented versions
                augmented_samples = apply_augmentations(audio, sr, NUM_AUGMENTATIONS)
                
                for i, aug_audio in enumerate(augmented_samples):
                    aug_filename = os.path.splitext(base_name)[0] + f"_aug{i+1}.mp3"
                    aug_output_path = os.path.join(train_dir, aug_filename)
                    sf.write(aug_output_path, aug_audio, sr)
                    train_files.append(aug_output_path)
                    
            except Exception as e:
                print(f"Error processing {train_file}: {e}")
        
        # Process test files (non-001 files)
        for test_file in test_candidates:
            try:
                # Load the audio file
                audio, sr = librosa.load(test_file, sr=None)
                
                # Create output filename
                base_name = os.path.basename(test_file)
                test_output_path = os.path.join(test_dir, base_name)
                
                # Save to test directory
                sf.write(test_output_path, audio, sr)
                test_files.append(test_output_path)
                
            except Exception as e:
                print(f"Error processing {test_file}: {e}")
    
    print(f"Processing complete. Generated {len(train_files)} training files and {len(test_files)} test files.")
    
    # Create a summary dataframe and save it
    summary = pd.DataFrame({
        'Species': [get_species_from_filename(f) for f in train_files + test_files],
        'Filename': [os.path.basename(f) for f in train_files + test_files],
        'Split': ['train'] * len(train_files) + ['test'] * len(test_files)
    })
    
    summary_path = os.path.join(output_base_dir, "dataset_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    process_audio_files()