import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle

# Path to directories containing processed audio files
base_dir = "/export/home/anandr/bird_proj/processed_audios/"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Path to save the mel spectrograms
output_dir = "/export/home/anandr/bird_proj/mel_spectrograms/"
train_mel_dir = os.path.join(output_dir, "train")
test_mel_dir = os.path.join(output_dir, "test")
plot_dir = os.path.join(output_dir, "plots")  # For visualization

# Create output directories
os.makedirs(train_mel_dir, exist_ok=True)
os.makedirs(test_mel_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Mel spectrogram parameters
SAMPLE_RATE = 48000  # Sample rate
n_fft = 2048  # FFT window size
hop_length = 512  # Hop length (step size)
n_mels = 128  # Number of mel bands

# Max duration in seconds (for padding/truncating)
max_duration = 10  # Adjust based on your data

def extract_species_from_filename(filename):
    """Extract species name from the filename"""
    base_name = os.path.basename(filename)
    
    # Handle augmented filenames
    if '_aug' in base_name:
        # Remove augmentation suffix first
        base_name = base_name.split('_aug')[0]
    
    # Now handle the sample number
    parts = base_name.split('_')
    if len(parts) > 1 and parts[-1].isdigit():
        # Return everything except the last part (sample number)
        return '_'.join(parts[:-1])
    
    # If the last part starts with a digit but isn't only digits
    if len(parts) > 1 and parts[-1][0].isdigit():
        # Try to separate the digits from the rest
        last_part = parts[-1]
        for i, char in enumerate(last_part):
            if not char.isdigit():
                # Split at the first non-digit
                parts[-1] = last_part[:i]
                break
    
    # Return joined parts
    return '_'.join([p for p in parts if p])

def generate_mel_spectrogram(audio_path, plot=False):
    """Generate mel spectrogram from audio file"""
    try:
        # Load audio file with the known sample rate
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Trim leading and trailing silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Pad or truncate to max_duration
        target_length = SAMPLE_RATE * max_duration
        if len(y) < target_length:
            # Pad with zeros
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            # Truncate
            y = y[:target_length]
        
        # Calculate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=SAMPLE_RATE,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=20,
            fmax=SAMPLE_RATE/2
        )
        
        # Convert to decibels
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to range [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Create a plot if requested
        if plot:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                mel_spec_db,
                sr=SAMPLE_RATE,
                hop_length=hop_length,
                x_axis='time',
                y_axis='mel',
                fmin=20,
                fmax=SAMPLE_RATE/2
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel spectrogram - {os.path.basename(audio_path)}')
            
            # Save the plot
            plot_filename = os.path.splitext(os.path.basename(audio_path))[0] + '.png'
            plt.savefig(os.path.join(plot_dir, plot_filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        return mel_spec_norm
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_files(split='train'):
    """Process all audio files in the specified split"""
    source_dir = train_dir if split == 'train' else test_dir
    target_dir = train_mel_dir if split == 'train' else test_mel_dir
    
    # Get all MP3 files
    audio_files = glob.glob(os.path.join(source_dir, "*.mp3"))
    print(f"Found {len(audio_files)} {split} audio files.")
    
    # Process each file
    processed_files = []
    features = []
    labels = []
    
    for audio_file in tqdm(audio_files, desc=f"Processing {split} files"):
        # Generate mel spectrogram
        mel_spec = generate_mel_spectrogram(
            audio_file, 
            plot=(len(processed_files) < 5)  # Plot first 5 examples only
        )
        
        if mel_spec is not None:
            # Save the mel spectrogram
            filename = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(target_dir, filename + '.npy')
            np.save(output_path, mel_spec)
            
            # Extract species name for labels
            species = extract_species_from_filename(audio_file)
            
            # Add to lists
            processed_files.append(output_path)
            features.append(mel_spec)
            labels.append(species)
    
    print(f"Successfully processed {len(processed_files)} {split} files.")
    
    # Return the data
    return {
        'files': processed_files,
        'features': features,
        'labels': labels
    }

def main():
    """Main function to process all data"""
    print("Generating mel spectrograms for training data...")
    train_data = process_audio_files('train')
    
    print("\nGenerating mel spectrograms for test data...")
    test_data = process_audio_files('test')
    
    # Save metadata
    metadata = {
        'train': {
            'files': train_data['files'],
            'labels': train_data['labels']
        },
        'test': {
            'files': test_data['files'],
            'labels': test_data['labels']
        },
        'params': {
            'sample_rate': SAMPLE_RATE,
            'n_fft': n_fft,
            'hop_length': hop_length,
            'n_mels': n_mels,
            'max_duration': max_duration
        },
        'unique_species': sorted(list(set(train_data['labels'] + test_data['labels'])))
    }
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    # Also save a readable CSV summary
    summary_df = pd.DataFrame({
        'Filename': [os.path.basename(f) for f in train_data['files'] + test_data['files']],
        'Species': train_data['labels'] + test_data['labels'],
        'Split': ['train'] * len(train_data['files']) + ['test'] * len(test_data['files'])
    })
    
    summary_df.to_csv(os.path.join(output_dir, 'spectrogram_summary.csv'), index=False)
    
    # Print summary statistics
    print("\nSummary:")
    print(f"Total training mel spectrograms: {len(train_data['files'])}")
    print(f"Total test mel spectrograms: {len(test_data['files'])}")
    print(f"Total unique species: {len(metadata['unique_species'])}")
    
    # Print class distribution
    train_species_counts = pd.Series(train_data['labels']).value_counts()
    test_species_counts = pd.Series(test_data['labels']).value_counts()
    
    print("\nTop 5 species by training sample count:")
    print(train_species_counts.head(5))

if __name__ == "__main__":
    main()