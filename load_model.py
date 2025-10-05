import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def create_genre_classifier(num_classes=16, pretrained=True, device=None):
    """
    Create a ResNet18 model pre-trained on ImageNet, modified for music genre classification.
    
    Parameters:
    -----------
    num_classes : int
        Number of output classes (default: 16 for music genres)
    pretrained : bool
        Whether to load ImageNet pre-trained weights (default: True)
    device : str or torch.device
        Device to move the model to ('cpu', 'cuda', etc.). If None, defaults to CPU.
    
    Returns:
    --------
    torch.nn.Module
        Modified ResNet18 model ready for genre classification
    """
    # Load pre-trained ResNet18
    if pretrained:
        print("Loading pre-trained ResNet18 from ImageNet...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        print("Creating ResNet18 without pre-trained weights...")
        model = models.resnet18(weights=None)
    
    # Get the number of features in the final layer
    num_features = model.fc.in_features
    
    # Replace the final fully connected layer for 16 classes
    model.fc = nn.Linear(num_features, num_classes)
    
    # Move model to specified device (CPU by default)
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    
    print(f"Model modified for {num_classes} classes and moved to {device}")
    print(f"Final layer: {model.fc}")
    
    return model


def load_trained_model(checkpoint_path, num_classes=16, device=None):
    """
    Load a trained model from a checkpoint file.
    
    Parameters:
    -----------
    checkpoint_path : str
        Path to the saved model checkpoint
    num_classes : int
        Number of output classes (default: 16)
    device : str or torch.device
        Device to move the model to. If None, defaults to CPU.
    
    Returns:
    --------
    torch.nn.Module
        Loaded model ready for inference
    """
    # Create the model architecture
    model = create_genre_classifier(num_classes=num_classes, pretrained=False, device=device)
    
    # Load the saved weights
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {checkpoint_path}")
    
    return model


class GenreClassifier(nn.Module):
    """
    Custom wrapper class for ResNet18-based genre classifier.
    Provides additional functionality for genre classification tasks.
    """
    def __init__(self, num_classes=16, pretrained=True):
        super(GenreClassifier, self).__init__()
        
        # Load base ResNet18 model
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet18(weights=None)
        
        # Get number of features from the original fc layer
        num_features = self.resnet.fc.in_features
        
        # Replace the final layer with custom classifier
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.resnet(x)
    
    def predict_proba(self, x):
        """Get probability distributions for predictions."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """Get predicted class indices."""
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Music Genre Classification Model")
    print("=" * 60)
    
    # Method 1: Simple function-based approach
    print("\nMethod 1: Creating model using function...")
    model = create_genre_classifier(num_classes=16, pretrained=True, device='cpu')
    
    # Print model summary
    print("\nModel architecture (last few layers):")
    print("..." )
    print(model.layer4)
    print(model.avgpool)
    print(model.fc)
    
    # Test with random input (simulating a 496x496 RGB spectrogram image)
    print("\n" + "=" * 60)
    print("Testing model with random input...")
    dummy_input = torch.randn(1, 3, 496, 496)  # Batch size 1, 3 channels, 496x496
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0][:5]}...")  # Show first 5 values
    
    # Get predicted class
    predicted_class = torch.argmax(output, dim=1)
    print(f"Predicted class: {predicted_class.item()}")
    
    # Method 2: Custom class-based approach
    print("\n" + "=" * 60)
    print("\nMethod 2: Creating model using custom class...")
    classifier = GenreClassifier(num_classes=16, pretrained=True)
    classifier = classifier.to('cpu')
    classifier.eval()
    
    # Test prediction methods
    with torch.no_grad():
        probs = classifier.predict_proba(dummy_input)
        pred_class = classifier.predict(dummy_input)
    
    print(f"Predicted class: {pred_class.item()}")
    print(f"Top 3 probabilities:")
    top3_probs, top3_indices = torch.topk(probs[0], 3)
    for i in range(3):
        print(f"  Class {top3_indices[i].item()}: {top3_probs[i].item():.4f}")
    
    print("\n" + "=" * 60)
    print("Model ready for training or inference!")
    print("=" * 60)
