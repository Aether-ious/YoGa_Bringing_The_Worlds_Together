import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import os

# --- MODEL DEFINITION ---
class YogaLSTM(nn.Module):
    def __init__(self, input_size=132, hidden_size=64, num_layers=2, num_classes=3):
        super(YogaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        return self.fc(out)

def train():
    print("🚀 Starting Training...")
    
    # 1. Load Data
    if not os.path.exists('model_data/X.npy'):
        print("❌ Data not found. Run 'src/extract_data.py' first.")
        return

    X = np.load('model_data/X.npy')
    y = np.load('model_data/y.npy')
    actions = np.load('model_data/labels.npy')
    
    print(f"📂 Loading data for classes: {actions}")
    
    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    
    # 2. Initialize Model
    num_classes = len(actions)
    model = YogaLSTM(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    num_epochs = 60
    print(f"🧠 Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
        if (epoch+1) % 10 == 0:
            print(f'   Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 4. Save
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/yoga_lstm.pth")
    print(f"\n✅ Model trained and saved to 'models/yoga_lstm.pth'")

if __name__ == "__main__":
    train()