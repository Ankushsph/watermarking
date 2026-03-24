"""
Improved CNN-LSTM Model with Attention and Residual Connections
For achieving 97%+ accuracy in dysarthric speech watermarking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

# Import config values
try:
    from .config import CNN_INPUT_SIZE, CNN_MODEL_PATH, SEQUENCE_LENGTH
except ImportError:
    from config import CNN_INPUT_SIZE, CNN_MODEL_PATH, SEQUENCE_LENGTH

# Use CNN_INPUT_SIZE as FRAME_SIZE
FRAME_SIZE = CNN_INPUT_SIZE

class AttentionLayer(nn.Module):
    """Self-attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        # attention_weights: (batch, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context: (batch, hidden_size)
        return context, attention_weights

class ResidualBlock(nn.Module):
    """Residual block for CNN"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class ImprovedHybridCNNLSTMWatermarker(nn.Module):
    """
    Improved CNN-LSTM model with:
    - Residual connections in CNN
    - Attention mechanism in LSTM
    - Batch normalization
    - Dropout for regularization
    - Deeper architecture
    """
    
    def __init__(self, input_size=FRAME_SIZE, hidden_size=256, num_layers=3, dropout=0.3):
        super(ImprovedHybridCNNLSTMWatermarker, self).__init__()
        
        # Improved CNN with residual blocks
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.res_block1 = ResidualBlock(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)
        
        self.res_block2 = ResidualBlock(128, 256, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)
        
        self.res_block3 = ResidualBlock(256, 512, kernel_size=3)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LSTM with more layers
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # *2 for bidirectional
        
        # Fully connected layers with residual
        self.fc1 = nn.Linear(hidden_size * 2, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 2)  # Binary classification
        
    def forward(self, x):
        # x shape: (batch, seq_len, frame_size)
        batch_size, seq_len, frame_size = x.shape
        
        # Process each frame through CNN
        x = x.view(batch_size * seq_len, 1, frame_size)
        
        # CNN with residual blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.res_block1(x)
        x = self.pool2(x)
        
        x = self.res_block2(x)
        x = self.pool3(x)
        
        x = self.res_block3(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # (batch*seq_len, 512)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        context, attention_weights = self.attention(lstm_out)
        
        # Fully connected with dropout
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(context))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        return x

def train_improved_model(train_loader, val_loader, num_epochs=100, learning_rate=0.0001):
    """
    Train improved CNN-LSTM model with:
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Better optimization
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Initialize improved model
    model = ImprovedHybridCNNLSTMWatermarker(
        input_size=FRAME_SIZE,
        hidden_size=256,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (frames, labels) in enumerate(train_loader):
            frames, labels = frames.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames, labels = frames.to(device), labels.to(device)
                outputs = model(frames)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.2f}%", end='')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CNN_MODEL_PATH)
            print(f" → Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            print()
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model

def load_improved_model():
    """Load the trained improved model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ImprovedHybridCNNLSTMWatermarker(
        input_size=FRAME_SIZE,
        hidden_size=256,
        num_layers=3,
        dropout=0.3
    ).to(device)
    
    if os.path.exists(CNN_MODEL_PATH):
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
        model.eval()
        print("Pre-trained improved model loaded successfully!")
    else:
        print("No pre-trained model found. Using untrained improved model.")
    
    return model
