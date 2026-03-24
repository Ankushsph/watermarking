# Step 4 & 5: CNN Feature Extraction + LSTM Temporal Modeling

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import *

class CNNFeatureExtractor(nn.Module):
    """
    Step 4: CNN-FE - Convolutional Neural Network for Feature Extraction
    Learns optimal embedding zones in dysarthric speech
    """
    def __init__(self, input_size=CNN_INPUT_SIZE):
        super(CNNFeatureExtractor, self).__init__()
        
        # Conv Block 1
        self.conv1 = nn.Conv1d(1, CNN_FILTERS[0], kernel_size=CNN_KERNEL_SIZE, padding=1)
        self.bn1 = nn.BatchNorm1d(CNN_FILTERS[0])
        self.pool1 = nn.MaxPool1d(CNN_POOL_SIZE)
        
        # Conv Block 2
        self.conv2 = nn.Conv1d(CNN_FILTERS[0], CNN_FILTERS[1], kernel_size=CNN_KERNEL_SIZE, padding=1)
        self.bn2 = nn.BatchNorm1d(CNN_FILTERS[1])
        self.pool2 = nn.MaxPool1d(CNN_POOL_SIZE)
        
        # Conv Block 3
        self.conv3 = nn.Conv1d(CNN_FILTERS[1], CNN_FILTERS[2], kernel_size=CNN_KERNEL_SIZE, padding=1)
        self.bn3 = nn.BatchNorm1d(CNN_FILTERS[2])
        self.pool3 = nn.MaxPool1d(CNN_POOL_SIZE)
        
        # Calculate output size after pooling
        self.feature_size = CNN_FILTERS[2] * (input_size // (CNN_POOL_SIZE ** 3))
        
        # Fully connected layer for embedding strength prediction
        self.fc = nn.Linear(self.feature_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch, 1, input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Predict embedding strength (0-1)
        embedding_strength = self.sigmoid(self.fc(x))
        
        return embedding_strength

class LSTMTemporalModel(nn.Module):
    """
    Step 5: LSTM-TM - Long Short-Term Memory for Temporal Modeling
    Models dysarthric speech patterns over time
    Handles irregular speech patterns
    """
    def __init__(self, input_size=CNN_FILTERS[2], hidden_size=LSTM_HIDDEN_SIZE):
        super(LSTMTemporalModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = LSTM_NUM_LAYERS
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism for dysarthric speech
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # Output layer for bit prediction
        self.fc = nn.Linear(hidden_size * 2, 2)  # Binary classification
        
    def forward(self, x, hidden=None):
        # x shape: (batch, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        # lstm_out shape: (batch, sequence_length, hidden_size * 2)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # attention_weights shape: (batch, sequence_length, 1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)
        # context shape: (batch, hidden_size * 2)
        
        # Predict bit
        output = self.fc(context)
        
        return output, hidden, attention_weights

class HybridCNNLSTMWatermarker(nn.Module):
    """
    Combined CNN-LSTM model for end-to-end watermarking
    """
    def __init__(self):
        super(HybridCNNLSTMWatermarker, self).__init__()
        
        self.cnn = CNNFeatureExtractor()
        self.lstm = LSTMTemporalModel()
        
    def forward(self, frames_sequence):
        # frames_sequence shape: (batch, sequence_length, frame_size)
        
        batch_size, seq_len, frame_size = frames_sequence.shape
        
        # Extract CNN features for each frame
        cnn_features = []
        embedding_strengths = []
        
        for i in range(seq_len):
            frame = frames_sequence[:, i, :].unsqueeze(1)  # (batch, 1, frame_size)
            strength = self.cnn(frame)
            embedding_strengths.append(strength)
            
            # Get intermediate features from CNN
            x = frame
            x = F.relu(self.cnn.bn1(self.cnn.conv1(x)))
            x = self.cnn.pool1(x)
            x = F.relu(self.cnn.bn2(self.cnn.conv2(x)))
            x = self.cnn.pool2(x)
            x = F.relu(self.cnn.bn3(self.cnn.conv3(x)))
            x = self.cnn.pool3(x)
            x = x.mean(dim=2)  # Global average pooling
            cnn_features.append(x)
        
        # Stack features
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, feature_dim)
        embedding_strengths = torch.stack(embedding_strengths, dim=1)  # (batch, seq_len, 1)
        
        # LSTM temporal modeling
        bit_prediction, _, attention_weights = self.lstm(cnn_features)
        
        return bit_prediction, embedding_strengths, attention_weights

def train_cnn_lstm_model(train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """
    Train the CNN-LSTM model for watermark embedding/extraction
    """
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = HybridCNNLSTMWatermarker().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            bit_pred, _, _ = model(frames)
            loss = criterion(bit_pred, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(bit_pred.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(device)
                labels = labels.to(device)
                
                bit_pred, _, _ = model(frames)
                loss = criterion(bit_pred, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(bit_pred.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {train_loss/len(train_loader):.4f} '
              f'Train Acc: {train_acc:.2f}% '
              f'Val Loss: {val_loss/len(val_loader):.4f} '
              f'Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CNN_MODEL_PATH.replace('.pth', '_best.pth'))
            print(f'  → Best model saved! Val Acc: {val_acc:.2f}%')
    
    return model

def load_trained_model():
    """Load pre-trained CNN-LSTM model"""
    device = torch.device('cuda' if USE_GPU and torch.cuda.is_available() else 'cpu')
    model = HybridCNNLSTMWatermarker().to(device)
    
    try:
        model.load_state_dict(torch.load(CNN_MODEL_PATH.replace('.pth', '_best.pth'), 
                                        map_location=device))
        model.eval()
        print("Pre-trained model loaded successfully!")
        return model
    except:
        print("No pre-trained model found. Using untrained model.")
        return model
