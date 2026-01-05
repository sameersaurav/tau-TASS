import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.interpolate import griddata

def load_valid_data(file):
    with open(file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  
    data = np.array([list(map(float, line.split())) for line in lines])
    return data

data = load_valid_data('free_energy.dat')

print("Data shape:", data.shape)  

if data.shape[0] != 10000:
    raise ValueError(f"Unexpected number of data points: {data.shape[0]}, expected 10000.")

xy_train, xy_val, F_train, F_val = train_test_split(data[:, :2], data[:, 2], test_size=0.2, random_state=0)

xy_train = torch.tensor(xy_train, dtype=torch.float32)
F_train = torch.tensor(F_train, dtype=torch.float32).view(-1, 1)
xy_val = torch.tensor(xy_val, dtype=torch.float32)
F_val = torch.tensor(F_val, dtype=torch.float32).view(-1, 1)

def af(x):
    return 1/(1+x**2)

class FreeEnergyNet(nn.Module):
    def __init__(self):
        super(FreeEnergyNet, self).__init__()
        self.fc1 = nn.Linear(2, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = af(self.fc1(x)) 
        x = af(self.fc2(x))
        x = self.fc3(x)
        return x

model = FreeEnergyNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
xy_train = xy_train.to(device)
F_train = F_train.to(device)
xy_val = xy_val.to(device)
F_val = F_val.to(device)

epochs = 50000
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(xy_train)
    loss = criterion(outputs, F_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(xy_val)
        val_loss = criterion(val_outputs, F_val)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

model.to('cpu')
xy_train = xy_train.to('cpu')
F_train = F_train.to('cpu')
xy_val = xy_val.to('cpu')
F_val = F_val.to('cpu')

with torch.no_grad():
    F_train_predicted = model(xy_train).numpy()

torch.save(model.state_dict(), 'free_energy_net.pt')
print("Model saved successfully.")

x_values_train = xy_train[:, 0].numpy()
y_values_train = xy_train[:, 1].numpy()
F_train_true = F_train.numpy().reshape(-1)  
F_train_pred = F_train_predicted.reshape(-1)  

grid_x, grid_y = np.mgrid[x_values_train.min():x_values_train.max():100j, y_values_train.min():y_values_train.max():100j]

F_train_true_grid = griddata((x_values_train, y_values_train), F_train_true, (grid_x, grid_y), method='linear')
F_train_pred_grid = griddata((x_values_train, y_values_train), F_train_pred, (grid_x, grid_y), method='linear')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
contour1 = plt.contourf(grid_x, grid_y, F_train_true_grid, cmap='viridis')
plt.colorbar(contour1)
plt.title('True Free Energy Surface (Training Data)')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
contour2 = plt.contourf(grid_x, grid_y, F_train_pred_grid, cmap='viridis')
plt.colorbar(contour2)
plt.title('Predicted Free Energy Surface (Training Data)')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.savefig('free_energy_surface.png')
plt.show()

true_surface = np.column_stack((grid_x.ravel(), grid_y.ravel(), F_train_true_grid.ravel()))
np.savetxt('true_free_energy_surface.dat', true_surface, header='x y F_true')
print("True free energy surface saved to 'true_free_energy_surface.dat'.")

pred_surface = np.column_stack((grid_x.ravel(), grid_y.ravel(), F_train_pred_grid.ravel()))
np.savetxt('predicted_train_free_energy_surface.dat', pred_surface, header='x y F_pred')
print("Predicted free energy surface saved to 'predicted_train_free_energy_surface.dat'.")

x = np.linspace(-np.pi, np.pi, 100)
y = np.linspace(-np.pi, np.pi, 100)
X, Y = np.meshgrid(x, y)

input_data = np.column_stack([X.ravel(), Y.ravel()])

input_tensor = torch.FloatTensor(input_data)

with torch.no_grad():  
    predicted_output = model(input_tensor)

predicted_output = predicted_output.numpy()
predicted_output_2d = predicted_output.reshape(100, 100)

with open('predicted_free_energy_surface.dat', 'w') as file:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            file.write(f"{X[i, j]} {Y[i, j]} {predicted_output_2d[i, j]}\n")

plt.figure(figsize=(6, 6))
plt.contourf(X, Y, predicted_output_2d, cmap='viridis')
plt.title('Predicted Free Energy Surface (PyTorch)')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')

plt.savefig('predicted_free_energy_contour_plot.png')

plt.show()

