import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import re
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the ProtoNet model (must match the architecture used for training)
class ProtoNetEmbedding(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=64):
        super(ProtoNetEmbedding, self).__init__()
        
        # Convolutional neural network for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # Flatten the output

def load_sample(file_path):
    """Load a single mel spectrogram sample"""
    # Load mel spectrogram
    spectrogram = np.load(file_path)
    
    # Convert to tensor
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
    
    # Add channel dimension if not present
    if len(spectrogram.shape) == 2:
        spectrogram = spectrogram.unsqueeze(0)
    
    # Add batch dimension
    spectrogram = spectrogram.unsqueeze(0)
    
    return spectrogram

def extract_species_name(filename):
    """Extract only the species name from the filename, removing sample numbers and augmentation markers"""
    # Remove file extension
    base_name = os.path.basename(filename)
    
    # Pattern to match: species name followed by _001, _002, etc. and optional _aug1, _aug2, etc.
    match = re.match(r'(.+?)_\d+(?:_aug\d+)?\.npy', base_name)
    
    if match:
        return match.group(1)  # Return just the species name
    else:
        # Fallback: remove extension and return
        return os.path.splitext(base_name)[0]

def get_class_examples(data_path, n_shot=1):
    """Get support examples for each class from the training set"""
    class_files = defaultdict(list)
    
    # Get all training files
    train_path = os.path.join(data_path, 'train')
    all_files = os.listdir(train_path)
    
    # Group files by species (not by filename pattern)
    for file_name in all_files:
        if file_name.endswith('.npy'):
            # Extract species name from file name (remove sample numbers and aug markers)
            species_name = extract_species_name(file_name)
            
            file_path = os.path.join(train_path, file_name)
            class_files[species_name].append(file_path)
    
    # Select n_shot examples for each class
    class_examples = {}
    for species_name, files in class_files.items():
        # Prefer non-augmented files for support examples
        original_files = [f for f in files if 'aug' not in os.path.basename(f)]
        
        if len(original_files) >= n_shot:
            selected_files = random.sample(original_files, n_shot)
        else:
            # If not enough original files, use augmented ones too
            selected_files = random.sample(files, min(n_shot, len(files)))
        
        class_examples[species_name] = selected_files
    
    return class_examples, list(class_files.keys())

def create_class_prototypes(model, class_examples, classes):
    """Create class prototypes from support examples"""
    model.eval()
    prototypes = {}
    
    with torch.no_grad():
        for i, class_name in enumerate(classes):
            files = class_examples[class_name]
            features = []
            
            for file_path in files:
                # Load and process the sample
                sample = load_sample(file_path)
                sample = sample.to(device)
                
                # Extract features
                feature = model(sample)
                features.append(feature)
            
            # Calculate prototype as the mean of features
            prototype = torch.cat(features).mean(0)
            prototypes[class_name] = prototype
    
    return prototypes

def predict_sample(model, sample_path, prototypes):
    """Predict the class of a single sample by comparing to prototypes"""
    model.eval()
    
    # Load and process the sample
    sample = load_sample(sample_path)
    sample = sample.to(device)
    
    with torch.no_grad():
        # Extract features
        query_feature = model(sample)
        
        # Calculate distances to prototypes
        distances = {}
        for class_name, prototype in prototypes.items():
            distance = F.pairwise_distance(query_feature, prototype.unsqueeze(0))
            distances[class_name] = distance.item()
        
        # Sort distances and get top predictions
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        predicted_class = sorted_distances[0][0]
        
        # Calculate confidence (using negative exponential of distance)
        confidence = np.exp(-sorted_distances[0][1])
        
        # Get all confidences for analysis
        confidences = {cls: np.exp(-dist) for cls, dist in distances.items()}
        sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        
        return predicted_class, confidence, sorted_confidences

def test_all_samples(model, test_dir, prototypes, classes):
    """Test all samples in the test directory and report results"""
    all_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]
    print(f"Found {len(all_files)} test files")
    
    results = []
    
    for file_name in tqdm(all_files, desc="Testing samples"):
        file_path = os.path.join(test_dir, file_name)
        true_species = extract_species_name(file_name)
        
        # Predict
        predicted_species, confidence, sorted_confidences = predict_sample(
            model, file_path, prototypes)
        
        # Record result
        correct = (predicted_species == true_species)
        
        results.append({
            'file_name': file_name,
            'true_species': true_species,
            'predicted_species': predicted_species,
            'confidence': confidence,
            'correct': correct,
            'top_confidences': sorted_confidences[:5]  # Top 5 predictions
        })
    
    return results

def analyze_confidence_vs_accuracy(results):
    """Analyze how accuracy changes with confidence threshold"""
    # Sort results by confidence
    sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
    
    # Calculate cumulative accuracy at different thresholds
    thresholds = []
    accuracies = []
    coverage_percentages = []
    
    total_samples = len(sorted_results)
    
    for i in range(1, total_samples + 1):
        # Consider top i samples by confidence
        subset = sorted_results[:i]
        
        # Calculate accuracy on this subset
        correct = sum(1 for r in subset if r['correct'])
        accuracy = correct / i
        
        # Record threshold and accuracy
        threshold = subset[-1]['confidence']
        coverage = i / total_samples * 100
        
        thresholds.append(threshold)
        accuracies.append(accuracy * 100)  # Convert to percentage
        coverage_percentages.append(coverage)
    
    # Create a confidence vs accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, accuracies)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Confidence Threshold')
    plt.grid(True)
    plt.savefig('accuracy_vs_confidence.png')
    
    # Create a coverage vs accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(coverage_percentages, accuracies)
    plt.xlabel('Coverage (%)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Coverage')
    plt.grid(True)
    plt.savefig('accuracy_vs_coverage.png')
    
    # Calculate accuracy at specific confidence thresholds
    confidence_levels = [0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.01]
    threshold_results = []
    
    for conf_threshold in confidence_levels:
        qualified_samples = [r for r in results if r['confidence'] >= conf_threshold]
        
        if qualified_samples:
            correct = sum(1 for r in qualified_samples if r['correct'])
            accuracy = (correct / len(qualified_samples)) * 100
            coverage = (len(qualified_samples) / total_samples) * 100
        else:
            accuracy = 0
            coverage = 0
        
        threshold_results.append({
            'threshold': conf_threshold,
            'accuracy': accuracy,
            'coverage': coverage,
            'samples': len(qualified_samples)
        })
    
    return threshold_results

def main():
    parser = argparse.ArgumentParser(description='Test ProtoNet on all bird audio samples')
    parser.add_argument('--model_path', type=str, default='best_protonet_hist.pth', help='Path to saved model')
    parser.add_argument('--data_path', type=str, default='mel_spectrograms', help='Path to data directory')
    parser.add_argument('--in_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of examples to use for each class prototype')
    
    args = parser.parse_args()
    
    # Load model
    model = ProtoNetEmbedding(in_channels=args.in_channels, embedding_dim=args.embedding_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")
    
    # Get support examples for each class
    class_examples, classes = get_class_examples(args.data_path, args.k_shot)
    print(f"Found {len(classes)} unique bird species")
    
    # Create class prototypes
    prototypes = create_class_prototypes(model, class_examples, classes)
    
    # Test all samples in the test directory
    test_dir = os.path.join(args.data_path, 'test')
    results = test_all_samples(model, test_dir, prototypes, classes)
    
    # Overall accuracy
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    overall_accuracy = (correct / total) * 100
    
    print(f"\nOverall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})")
    
    # Analyze confidence vs accuracy
    threshold_results = analyze_confidence_vs_accuracy(results)
    
    # Print accuracy at different confidence thresholds
    print("\nAccuracy at different confidence thresholds:")
    print("Threshold | Accuracy | Coverage | Samples")
    print("-" * 50)
    
    for r in threshold_results:
        print(f"{r['threshold']:.2f}      | {r['accuracy']:.2f}%   | {r['coverage']:.2f}%  | {r['samples']}/{total}")
    
    # Save results to CSV
    with open('test_results.csv', 'w') as f:
        f.write('file_name,true_species,predicted_species,confidence,correct\n')
        for r in results:
            f.write(f"{r['file_name']},{r['true_species']},{r['predicted_species']},{r['confidence']:.4f},{r['correct']}\n")
    
    print("\nDetailed results saved to test_results.csv")
    
    # Per-class analysis
    class_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for r in results:
        species = r['true_species']
        class_results[species]['total'] += 1
        if r['correct']:
            class_results[species]['correct'] += 1
    
    # Calculate per-class accuracy
    for species in class_results:
        correct = class_results[species]['correct']
        total = class_results[species]['total']
        accuracy = (correct / total) * 100
        class_results[species]['accuracy'] = accuracy
    
    # Sort classes by accuracy
    sorted_classes = sorted(class_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    # Save per-class results
    with open('class_results.csv', 'w') as f:
        f.write('species,accuracy,correct,total\n')
        for species, stats in sorted_classes:
            f.write(f"{species},{stats['accuracy']:.2f},{stats['correct']},{stats['total']}\n")
    
    print("\nPer-class results saved to class_results.csv")
    
    # Plot distribution of per-class accuracies
    accuracies = [stats['accuracy'] for _, stats in sorted_classes]
    
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=10, edgecolor='black')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Number of Species')
    plt.title('Distribution of Per-Species Accuracies')
    plt.savefig('accuracy_distribution.png')
    
    # Create confusion matrix for most confused classes
    confusion_counts = defaultdict(int)
    
    for r in results:
        if not r['correct']:
            confusion_pair = (r['true_species'], r['predicted_species'])
            confusion_counts[confusion_pair] += 1
    
    # Sort by confusion count
    sorted_confusion = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Save most confused pairs
    with open('confusion_pairs.csv', 'w') as f:
        f.write('true_species,predicted_species,count\n')
        for (true_species, pred_species), count in sorted_confusion[:20]:  # Top 20 confused pairs
            f.write(f"{true_species},{pred_species},{count}\n")
    
    print("\nTop confused species pairs saved to confusion_pairs.csv")
    print("\nAnalysis complete! Check the generated CSV files and plots for detailed results.")

if __name__ == "__main__":
    main()