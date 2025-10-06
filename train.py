"""
Training script for Music Genre Classification Model
Trains ResNet18 on ccmusic-database/music_genre dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
from datetime import datetime

# Import custom modules
from load_model import create_genre_classifier
from genre_labels import GENRE_LABELS, NUM_CLASSES

# Try to import datasets library
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


def check_gpu_availability():
    """Check and print GPU information."""
    print("\n" + "="*60)
    print("Device Configuration Check")
    print("="*60)
    
    print(f"PyTorch version: {torch.__version__}")
    print("Using CPU for training")
    return False


def optimize_gpu_settings():
    """Optimize PyTorch settings for CPU usage."""
    print("\n⚠️ Running on CPU - no optimizations applied")


class MusicGenreDataset(Dataset):
    """
    PyTorch Dataset for Music Genre Classification
    Uses CQT spectrograms from the ccmusic-database/music_genre dataset
    """
    
    def __init__(self, hf_dataset, transform=None, max_samples=None):
        """
        Parameters:
        -----------
        hf_dataset : datasets.Dataset
            Hugging Face dataset
        transform : callable, optional
            Transform to apply to images
        max_samples : int, optional
            Maximum number of samples to use (for faster training)
        """
        self.dataset = hf_dataset
        self.transform = transform
        
        # Limit dataset size if specified
        if max_samples and len(self.dataset) > max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        
        print(f"Dataset initialized with {len(self.dataset)} samples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
        --------
        tuple: (image_tensor, label)
        """
        item = self.dataset[idx]
        
        # Get CQT spectrogram (already preprocessed in the dataset)
        cqt = item['cqt']
        
        # Convert to PIL Image if it's a numpy array
        if isinstance(cqt, np.ndarray):
            # Normalize to 0-255 range if needed
            if cqt.max() <= 1.0:
                cqt = (cqt * 255).astype(np.uint8)
            elif cqt.min() < 0:
                # If values are in dB scale, normalize
                cqt = ((cqt - cqt.min()) / (cqt.max() - cqt.min()) * 255).astype(np.uint8)
            
            # Create PIL Image - ensure it's in the right shape
            if len(cqt.shape) == 2:
                # Grayscale, convert to RGB
                image = Image.fromarray(cqt).convert('RGB')
            elif len(cqt.shape) == 3:
                # Already has channels
                if cqt.shape[0] in [1, 3]:  # Channels first
                    cqt = np.transpose(cqt, (1, 2, 0))
                image = Image.fromarray(cqt.astype(np.uint8)).convert('RGB')
            else:
                raise ValueError(f"Unexpected CQT shape: {cqt.shape}")
        else:
            # Assume it's already a PIL Image
            image = cqt.convert('RGB')
        
        # Resize to 496x496
        image = image.resize((496, 496), Image.Resampling.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # Get label
        label = item['thr_level_label']
        
        return img_tensor, label


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=3, device='cpu'):
    """
    Train the model.
    
    Parameters:
    -----------
    model : nn.Module
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    criterion : nn.Module
        Loss function
    optimizer : optim.Optimizer
        Optimizer
    scheduler : optim.lr_scheduler, optional
        Learning rate scheduler
    num_epochs : int
        Number of training epochs
    device : str
        Device to train on
    
    Returns:
    --------
    dict: Training history
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}")
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Update learning rate based on validation accuracy
        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'genre_model_best.pth')
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    return history


def main():
    """Main training function."""
    print("="*60)
    print("Music Genre Classification - Training Script")
    print("="*60)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Check if datasets library is available
    if not DATASETS_AVAILABLE:
        print("\n❌ Error: 'datasets' library not found.")
        print("Install it with: pip install datasets")
        return
    
    # Configuration
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    LEARNING_RATE = 0.001
    MAX_TRAIN_SAMPLES = 1000
    MAX_VAL_SAMPLES = 200
    DEVICE = 'cpu'
    
    # Check device availability
    gpu_available = check_gpu_availability()
    
    # Optimize settings
    optimize_gpu_settings()
    
    print(f"\n📋 Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Max Train Samples: {MAX_TRAIN_SAMPLES}")
    print(f"  Max Val Samples: {MAX_VAL_SAMPLES}")
    
    # Load dataset
    print(f"\n📦 Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset("ccmusic-database/music_genre", name="eval")
        print(f"✅ Dataset loaded successfully!")
        print(f"  Train samples: {len(dataset['train'])}")
        print(f"  Test samples: {len(dataset['test'])}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("\nTrying to create synthetic dataset for testing...")
        from torch.utils.data import TensorDataset
        
        num_samples = MAX_TRAIN_SAMPLES
        synthetic_images = torch.randn(num_samples, 3, 496, 496)
        synthetic_labels = torch.randint(0, NUM_CLASSES, (num_samples,))
        
        train_dataset = TensorDataset(synthetic_images[:800], synthetic_labels[:800])
        val_dataset = TensorDataset(synthetic_images[800:], synthetic_labels[800:])
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"✅ Synthetic dataset created for testing")
        use_synthetic = True
    else:
        use_synthetic = False
    
    if not use_synthetic:
        # Create PyTorch datasets
        print(f"\n🔄 Creating PyTorch datasets...")
        train_dataset = MusicGenreDataset(
            dataset['train'],
            max_samples=MAX_TRAIN_SAMPLES
        )
        val_dataset = MusicGenreDataset(
            dataset['test'],
            max_samples=MAX_VAL_SAMPLES
        )
        
        # Create data loaders
        print(f"\n🔄 Creating data loaders...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
    
    print(f"✅ Data loaders created!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n🧠 Creating model...")
    model = create_genre_classifier(num_classes=NUM_CLASSES, pretrained=True, device=DEVICE)
    print(f"✅ Model created and moved to {DEVICE}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    print(f"\n📊 Loss function: CrossEntropyLoss")
    print(f"⚙️ Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"📉 LR Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    
    # Train model
    print(f"\n🚀 Starting training...")
    start_time = datetime.now()
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    # Save final model
    print(f"\n💾 Saving final model...")
    torch.save(model.state_dict(), 'genre_model.pth')
    print(f"✅ Model saved as 'genre_model.pth'")
    
    # Save training history
    print(f"\n💾 Saving training history...")
    torch.save(history, 'training_history.pth')
    print(f"✅ Training history saved as 'training_history.pth'")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Val Accuracy:   {history['val_acc'][-1]:.2f}%")
    print(f"\nBest Validation Accuracy: {max(history['val_acc']):.2f}%")
    print(f"\nSaved files:")
    print(f"  - genre_model.pth (final model)")
    print(f"  - genre_model_best.pth (best model)")
    print(f"  - training_history.pth (training history)")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
