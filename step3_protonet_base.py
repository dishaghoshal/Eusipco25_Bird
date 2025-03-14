import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict
import random
import time
import json
import csv
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

# Hyperparameters
N_WAY = 5  # Number of classes per episode
K_SHOT = 1  # Number of support examples per class
Q_QUERY = 5  # Number of query examples per class
EPISODES = 100  # Number of episodes for training
TEST_EPISODES = 10  # Number of episodes for testing
LR = 0.001  # Learning rate
EMBEDDING_DIM = 64  # Embedding dimension
EPOCHS = 30  # Number of epochs

# Create output directory for metrics
os.makedirs('metrics', exist_ok=True)

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

# Dataset class for bird mel spectrograms
class BirdSpectrogramDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        self.split = split
        
        # Get all file paths
        self.file_paths = []
        self.labels = []
        self.class_map = {}
        
        # Get all unique classes (species names)
        all_files = os.listdir(os.path.join(data_path, split))
        all_species = set()
        
        for file_name in all_files:
            if file_name.endswith('.npy'):
                # Extract species name from file name
                species_name = extract_species_name(file_name)
                all_species.add(species_name)
        
        # Create class to index mapping
        self.class_map = {species_name: idx for idx, species_name in enumerate(sorted(list(all_species)))}
        
        # Organize files by class
        self.class_files = defaultdict(list)
        
        for file_name in all_files:
            if file_name.endswith('.npy'):
                species_name = extract_species_name(file_name)
                file_path = os.path.join(data_path, split, file_name)
                self.class_files[species_name].append(file_path)
                self.file_paths.append(file_path)
                self.labels.append(self.class_map[species_name])
        
        print(f"Found {len(self.class_files)} unique bird species in {split} set")
        print(f"Total {len(self.file_paths)} files in {split} set")
        
        # Save dataset statistics
        self.save_dataset_stats()
    
    def save_dataset_stats(self):
        """Save dataset statistics to a file"""
        stats = {
            'split': self.split,
            'total_files': len(self.file_paths),
            'unique_species': len(self.class_files),
            'files_per_species': {species: len(files) for species, files in self.class_files.items()}
        }
        
        with open(f'metrics/dataset_stats_{self.split}.json', 'w') as f:
            json.dump(stats, f, indent=4)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load mel spectrogram
        spectrogram = np.load(file_path)
        
        # Convert to tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        
        # Add channel dimension if not present
        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        
        return spectrogram, label
    
    def get_class_count(self):
        return len(self.class_map)
    
    def get_class_files(self):
        return self.class_files
    
    def get_class_map(self):
        return self.class_map

# Create ProtoNet model
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

# Episode sampler for N-way, K-shot learning
class EpisodeSampler:
    def __init__(self, dataset, n_way, k_shot, q_query, episodes):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        
        self.class_files = dataset.get_class_files()
        self.classes = list(self.class_files.keys())
        
    def __len__(self):
        return self.episodes
    
    def __iter__(self):
        for _ in range(self.episodes):
            # Randomly select N classes
            selected_classes = random.sample(self.classes, self.n_way)
            
            support_files = []
            query_files = []
            support_labels = []
            query_labels = []
            
            # For each selected class
            for label, class_name in enumerate(selected_classes):
                class_files = self.class_files[class_name]
                
                # Ensure we have enough files for support and query
                if len(class_files) < self.k_shot + self.q_query:
                    # If not enough files, use augmented versions for query
                    original_files = [f for f in class_files if 'aug' not in f]
                    augmented_files = [f for f in class_files if 'aug' in f]
                    
                    # Select support files from original if possible
                    if len(original_files) >= self.k_shot:
                        selected_support = random.sample(original_files, self.k_shot)
                    else:
                        # Not enough original files, use what we have
                        selected_support = random.sample(class_files, min(self.k_shot, len(class_files)))
                    
                    remaining_files = [f for f in class_files if f not in selected_support]
                    
                    # Use remaining files for query
                    if remaining_files and len(remaining_files) >= self.q_query:
                        selected_query = random.sample(remaining_files, self.q_query)
                    else:
                        # If not enough remaining files or empty list, reuse support files with replacement
                        # This is crucial when we have very few examples per class
                        all_available = selected_support + remaining_files
                        selected_query = random.choices(all_available, k=self.q_query)
                else:
                    # Randomly sample files without replacement
                    selected_files = random.sample(class_files, self.k_shot + self.q_query)
                    selected_support = selected_files[:self.k_shot]
                    selected_query = selected_files[self.k_shot:]
                
                support_files.extend(selected_support)
                query_files.extend(selected_query)
                support_labels.extend([label] * self.k_shot)
                query_labels.extend([label] * self.q_query)
            
            # Load spectrograms for support and query sets
            support_spectrograms = []
            query_spectrograms = []
            
            for file_path in support_files:
                spectrogram = np.load(file_path)
                spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
                if len(spectrogram.shape) == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                support_spectrograms.append(spectrogram)
            
            for file_path in query_files:
                spectrogram = np.load(file_path)
                spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
                if len(spectrogram.shape) == 2:
                    spectrogram = spectrogram.unsqueeze(0)
                query_spectrograms.append(spectrogram)
            
            # Convert to tensors
            support_spectrograms = torch.stack(support_spectrograms)
            query_spectrograms = torch.stack(query_spectrograms)
            support_labels = torch.tensor(support_labels)
            query_labels = torch.tensor(query_labels)
            
            yield support_spectrograms, support_labels, query_spectrograms, query_labels, selected_classes

# Save metrics to CSV file
def save_metrics_to_csv(metrics, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['epoch'] + list(metrics.keys()))
        
        # Write data for each epoch
        for epoch in range(len(next(iter(metrics.values())))):
            row = [epoch+1]
            for metric in metrics.keys():
                row.append(metrics[metric][epoch])
            writer.writerow(row)

# Save episode metrics to CSV file
def save_episode_metrics(metrics, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['episode', 'accuracy', 'loss', 'classes_used'])
        
        # Write data for each episode
        for i, (acc, loss, classes) in enumerate(zip(metrics['accuracies'], metrics['losses'], metrics['classes'])):
            writer.writerow([i+1, acc, loss, ','.join(classes)])

# Calculate and save per-class metrics
def save_class_metrics(all_classes, all_predictions, all_labels, filename):
    # Ensure we're working with numeric labels
    if isinstance(all_labels[0], str):
        class_to_idx = {cls: i for i, cls in enumerate(set(all_classes))}
        predictions = [class_to_idx[cls] for cls in all_predictions]
        labels = [class_to_idx[cls] for cls in all_labels]
    else:
        predictions = all_predictions
        labels = all_labels
    
    # Get unique classes actually present in the data
    unique_classes = sorted(list(set(labels)))
    
    # Calculate per-class metrics using only the classes present in the data
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=unique_classes, zero_division=0
    )
    
    # Create confusion matrix
    cm = confusion_matrix(labels, predictions, labels=unique_classes)
    
    # Create a mapping between unique_classes and all_classes if needed
    class_map = {}
    if isinstance(all_classes[0], str):
        # If all_classes contains the actual class names, map numeric indices to names
        rev_class_to_idx = {i: cls for cls, i in class_to_idx.items()}
        for i, cls_idx in enumerate(unique_classes):
            if cls_idx in rev_class_to_idx:
                class_map[i] = rev_class_to_idx[cls_idx]
            else:
                class_map[i] = f"Class_{cls_idx}"
    else:
        # If all_classes is already numeric, just use the unique classes
        for i, cls_idx in enumerate(unique_classes):
            class_map[i] = all_classes[cls_idx] if cls_idx < len(all_classes) else f"Class_{cls_idx}"
    
    # Save per-class metrics
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['class', 'precision', 'recall', 'f1', 'support'])
        
        # Write data for each class
        for i in range(len(unique_classes)):
            class_name = class_map.get(i, f"Class_{unique_classes[i]}")
            writer.writerow([class_name, precision[i], recall[i], f1[i], support[i]])
    
    # Save confusion matrix
    np.save(filename.replace('.csv', '_confusion_matrix.npy'), cm)
    
    # Also save a simple text version
    with open(filename.replace('.csv', '_confusion_matrix.txt'), 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write("\n".join([" ".join([f"{x:5d}" for x in row]) for row in cm]))
    
    return precision, recall, f1, support

# ProtoNet training function with comprehensive metrics
def train_protonet(model, train_sampler, val_sampler, optimizer, epochs):
    # Create metrics dictionary
    metrics = {
        'train_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'train_time_per_epoch': [],
        'val_time_per_epoch': []
    }
    
    # Create episode-level metrics
    train_episode_metrics = {
        'accuracies': [],
        'losses': [],
        'classes': []
    }
    
    val_episode_metrics = {
        'accuracies': [],
        'losses': [],
        'classes': []
    }
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        epoch_start_time = time.time()
        
        progress_bar = tqdm(train_sampler, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for support_images, support_labels, query_images, query_labels, selected_classes in progress_bar:
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Extract features
            support_features = model(support_images)
            query_features = model(query_images)
            
            # Calculate prototypes
            unique_labels = torch.unique(support_labels)
            prototypes = torch.zeros(len(unique_labels), support_features.shape[1]).to(device)
            
            # Calculate prototype for each class
            for i, label in enumerate(unique_labels):
                mask = (support_labels == label)
                prototypes[i] = support_features[mask].mean(0)
            
            # Calculate distances between query features and prototypes
            dists = torch.cdist(query_features, prototypes)
            
            # Convert distances to log probabilities
            log_p_y = F.log_softmax(-dists, dim=1)
            
            # Calculate loss
            loss = F.nll_loss(log_p_y, query_labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy for this episode
            _, predicted = torch.min(dists, 1)
            predicted_labels = torch.tensor([unique_labels[p.item()] for p in predicted], device=device)
            episode_accuracy = 100 * (predicted_labels == query_labels).sum().item() / query_labels.size(0)
            
            # Store episode metrics
            train_episode_metrics['accuracies'].append(episode_accuracy)
            train_episode_metrics['losses'].append(loss.item())
            train_episode_metrics['classes'].append(selected_classes)
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'acc': episode_accuracy})
        
        epoch_train_time = time.time() - epoch_start_time
        metrics['train_time_per_epoch'].append(epoch_train_time)
        
        avg_train_loss = train_loss / len(train_sampler)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_start_time = time.time()
        correct = 0
        total = 0
        val_loss = 0
        
        # Store all predictions and labels for metrics calculation
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_sampler, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for support_images, support_labels, query_images, query_labels, selected_classes in progress_bar:
                support_images = support_images.to(device)
                query_images = query_images.to(device)
                support_labels = support_labels.to(device)
                query_labels = query_labels.to(device)
                
                # Extract features
                support_features = model(support_images)
                query_features = model(query_images)
                
                # Calculate prototypes
                unique_labels = torch.unique(support_labels)
                prototypes = torch.zeros(len(unique_labels), support_features.shape[1]).to(device)
                
                # Calculate prototype for each class
                for i, label in enumerate(unique_labels):
                    mask = (support_labels == label)
                    prototypes[i] = support_features[mask].mean(0)
                
                # Calculate distances between query features and prototypes
                dists = torch.cdist(query_features, prototypes)
                
                # Calculate validation loss
                log_p_y = F.log_softmax(-dists, dim=1)
                loss = F.nll_loss(log_p_y, query_labels)
                val_loss += loss.item()
                
                # Get predicted classes
                _, predicted = torch.min(dists, 1)
                
                # Map predictions back to original labels
                predicted_labels = torch.tensor([unique_labels[p.item()] for p in predicted], device=device)
                
                # Calculate episode accuracy
                episode_accuracy = 100 * (predicted_labels == query_labels).sum().item() / query_labels.size(0)
                
                # Store episode metrics
                val_episode_metrics['accuracies'].append(episode_accuracy)
                val_episode_metrics['losses'].append(loss.item())
                val_episode_metrics['classes'].append(selected_classes)
                
                # Update statistics
                correct += (predicted_labels == query_labels).sum().item()
                total += query_labels.size(0)
                
                # Store predictions and labels for metrics
                all_predictions.extend(predicted_labels.cpu().numpy())
                all_labels.extend(query_labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'accuracy': episode_accuracy, 'loss': loss.item()})
        
        epoch_val_time = time.time() - val_start_time
        metrics['val_time_per_epoch'].append(epoch_val_time)
        
        # Calculate validation metrics
        val_accuracy = 100 * correct / total
        metrics['val_accuracy'].append(val_accuracy)
        
        # Calculate precision, recall, F1 for the epoch
        unique_classes = list(set([int(label) for label in all_labels]))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, labels=unique_classes, average='macro', zero_division=0
        )
        
        metrics['val_precision'].append(precision)
        metrics['val_recall'].append(recall)
        metrics['val_f1'].append(f1)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        print(f'Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}')
        print(f'Train Time: {epoch_train_time:.2f}s, Val Time: {epoch_val_time:.2f}s')
        
        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_accuracy,
                'metrics': metrics
            }, 'metrics/best_protonet.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')
    
    # Save all metrics
    save_metrics_to_csv(metrics, 'metrics/training_metrics.csv')
    save_episode_metrics(train_episode_metrics, 'metrics/train_episode_metrics.csv')
    save_episode_metrics(val_episode_metrics, 'metrics/val_episode_metrics.csv')
    
    # Save hyperparameters to JSON
    hyperparams = {
        'n_way': N_WAY,
        'k_shot': K_SHOT,
        'q_query': Q_QUERY,
        'episodes': EPISODES,
        'test_episodes': TEST_EPISODES,
        'lr': LR,
        'embedding_dim': EMBEDDING_DIM,
        'epochs': EPOCHS,
        'seed': SEED,
        'device': str(device)
    }
    
    with open('metrics/hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    return metrics

# Evaluate the model on test set with detailed metrics
def evaluate_protonet(model, test_sampler, reverse_class_map=None):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_raw_classes = []
    
    # Store episode-level metrics
    episode_metrics = {
        'accuracies': [],
        'inference_times': [],
        'classes': []
    }
    
    with torch.no_grad():
        progress_bar = tqdm(test_sampler, desc='Evaluating on test set')
        for support_images, support_labels, query_images, query_labels, classes in progress_bar:
            episode_start_time = time.time()
            
            support_images = support_images.to(device)
            query_images = query_images.to(device)
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)
            
            # Extract features
            support_features = model(support_images)
            query_features = model(query_images)
            
            # Calculate prototypes
            unique_labels = torch.unique(support_labels)
            prototypes = torch.zeros(len(unique_labels), support_features.shape[1]).to(device)
            
            # Calculate prototype for each class
            for i, label in enumerate(unique_labels):
                mask = (support_labels == label)
                prototypes[i] = support_features[mask].mean(0)
            
            # Calculate distances between query features and prototypes
            dists = torch.cdist(query_features, prototypes)
            
            # Get predicted classes
            _, predicted = torch.min(dists, 1)
            
            # Map predictions back to original labels
            predicted_labels = torch.tensor([unique_labels[p.item()] for p in predicted], device=device)
            
            # Record inference time
            inference_time = time.time() - episode_start_time
            
            # Calculate episode accuracy
            episode_accuracy = 100 * (predicted_labels == query_labels).sum().item() / query_labels.size(0)
            
            # Store episode metrics
            episode_metrics['accuracies'].append(episode_accuracy)
            episode_metrics['inference_times'].append(inference_time)
            episode_metrics['classes'].append(classes)
            
            # Update statistics
            correct += (predicted_labels == query_labels).sum().item()
            total += query_labels.size(0)
            
            # Store predictions and labels
            all_predictions.extend(predicted_labels.cpu().numpy())
            all_labels.extend(query_labels.cpu().numpy())
            
            # Collect all class names used in this episode
            if reverse_class_map:
                all_raw_classes.extend(classes)
            
            # Update progress bar
            progress_bar.set_postfix({'accuracy': episode_accuracy, 'time': inference_time})
    
    # Calculate overall test metrics
    test_accuracy = 100 * correct / total
    
    # Calculate and save per-class metrics if we have class mapping
    if reverse_class_map:
        # Get the unique classes that were actually used in the test episodes
        unique_classes = sorted(list(set(all_raw_classes)))
        
        # For per-class metrics, we need to ensure consistency between predictions and labels
        # We'll create a list of label indices that were actually used
        used_label_indices = sorted(list(set(all_labels)))
        
        # Map these to class names if we have the mapping
        class_names = []
        for idx in used_label_indices:
            if idx in reverse_class_map:
                class_names.append(reverse_class_map[idx])
            else:
                class_names.append(f"Class_{idx}")
        
        # Save class metrics using the labels that were actually used
        _, _, _, _ = save_class_metrics(
            class_names,
            all_predictions, 
            all_labels, 
            'metrics/test_class_metrics.csv'
        )
    
    # Save episode metrics
    save_episode_metrics(episode_metrics, 'metrics/test_episode_metrics.csv')
    
    # Save overall test metrics
    test_metrics = {
        'accuracy': test_accuracy,
        'total_samples': total,
        'total_correct': correct,
        'avg_inference_time': sum(episode_metrics['inference_times']) / len(episode_metrics['inference_times'])
    }
    
    with open('metrics/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Average Inference Time: {test_metrics["avg_inference_time"]:.4f}s per episode')
    
    return test_accuracy, all_predictions, all_labels

# Main execution
def main():
    # Paths
    data_path = 'mel_spectrograms'
    
    # Track overall execution time
    total_start_time = time.time()
    
    # Create datasets
    train_dataset = BirdSpectrogramDataset(data_path, 'train')
    test_dataset = BirdSpectrogramDataset(data_path, 'test')
    
    # Check if test set has all the classes in train set
    train_classes = set(train_dataset.get_class_map().keys())
    test_classes = set(test_dataset.get_class_map().keys())
    
    print(f"Number of species in train set: {len(train_classes)}")
    print(f"Number of species in test set: {len(test_classes)}")
    
    # Get a sample to check the shape
    sample_spectrogram, _ = train_dataset[0]
    print(f"Sample spectrogram shape: {sample_spectrogram.shape}")
    
    # Create episodic samplers
    train_sampler = EpisodeSampler(train_dataset, N_WAY, K_SHOT, Q_QUERY, EPISODES)
    val_sampler = EpisodeSampler(test_dataset, N_WAY, K_SHOT, Q_QUERY, TEST_EPISODES)
    
    # Create model
    in_channels = sample_spectrogram.shape[0]  # Number of input channels (1 for grayscale spectrograms)
    model = ProtoNetEmbedding(in_channels=in_channels, embedding_dim=EMBEDDING_DIM).to(device)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=LR)
    
    # Report model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_structure': str(model)
    }
    
    with open('metrics/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # Train the model
    print("Starting training...")
    train_start_time = time.time()
    metrics = train_protonet(model, train_sampler, val_sampler, optimizer, EPOCHS)
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Load the best model
    checkpoint = torch.load('metrics/best_protonet.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_start_time = time.time()
    test_sampler = EpisodeSampler(test_dataset, N_WAY, K_SHOT, Q_QUERY, TEST_EPISODES)
    
    # Get reverse class map for mapping indices back to class names
    reverse_class_map = {idx: cls for cls, idx in test_dataset.get_class_map().items()}
    
    test_accuracy, all_predictions, all_labels = evaluate_protonet(model, test_sampler, reverse_class_map)
    test_time = time.time() - test_start_time
    
    total_time = time.time() - total_start_time
    
    # Save timing information
    timing_info = {
        'total_execution_time': total_time,
        'training_time': train_time,
        'testing_time': test_time,
        'time_per_epoch': train_time / EPOCHS
    }
    
    with open('metrics/timing_info.json', 'w') as f:
        json.dump(timing_info, f, indent=4)
    
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Total execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()