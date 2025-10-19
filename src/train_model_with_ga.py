# src/train_model_with_ga.py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create output directory for models
os.makedirs("models", exist_ok=True)

# Load the optimized features from genetic algorithm
print("Loading genetically optimized model...")
with open('models/ga_optimized_model.pkl', 'rb') as f:
    _, scaler, selected_features, emotions = pickle.load(f)

# Load the original data
with open('data/features/audio_features.pkl', 'rb') as f:
    X, y = pickle.load(f)

# Convert emotions to numerical labels
emotion_to_label = {emotion: i for i, emotion in enumerate(emotions)}
y_encoded = np.array([emotion_to_label[emotion] for emotion in y])

# Standardize features and select only those chosen by genetic algorithm
X_scaled = scaler.transform(X)
X_optimized = X_scaled[:, selected_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_optimized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
print(f"Using {len(selected_features)} features selected by genetic algorithm")

# Train a Random Forest with optimized features
print("\n--- Training Random Forest with Genetically Optimized Features ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy with GA Features: {accuracy_rf:.4f}")

# Print detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=emotions))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.title('Confusion Matrix (Random Forest with GA Features)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix_rf_ga.png')

# Now train a neural network using PyTorch
print("\n--- Training Neural Network with Genetically Optimized Features ---")

# Define the neural network
class EmotionNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(EmotionNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]  # This is now the number of selected features
hidden_size = 128
num_classes = len(emotions)
model = EmotionNN(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    train_losses.append(epoch_loss)
    test_accuracies.append(epoch_acc)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {epoch_acc:.4f}')

# Plot training progress
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss (GA Features)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.title('Test Accuracy (GA Features)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('models/training_progress_ga.png')

# Final evaluation on test set
model.eval()
y_pred_nn = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_nn.extend(predicted.numpy())
        y_true.extend(labels.numpy())

# Accuracy
accuracy_nn = accuracy_score(y_true, y_pred_nn)
print(f"\nNeural Network Final Accuracy with GA Features: {accuracy_nn:.4f}")

# Classification report
print("\nNeural Network Classification Report:")
print(classification_report(y_true, y_pred_nn, target_names=emotions))

# Confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred_nn)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotions, yticklabels=emotions)
plt.title('Confusion Matrix (Neural Network with GA Features)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix_nn_ga.png')

# Save the models
with open('models/rf_model_with_ga.pkl', 'wb') as f:
    pickle.dump((rf_model, scaler, selected_features, emotions), f)

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'selected_features': selected_features,
    'emotions': emotions
}, 'models/nn_model_with_ga.pt')

print("\nModels with genetically optimized features saved successfully!")