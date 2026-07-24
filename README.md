# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY

A Neural Network is a machine learning model that learns the relationship between input and output data. For regression problems, it predicts continuous numeric values by adjusting its weights during training to reduce prediction error.

## Neural Network Model


<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/92c3d567-21e0-4760-b6ee-e6bdebd1acb8" />



## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Harish R

### Register Number: 212224230085

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\Akbar\Downloads\deep learning exp\Exp-1.csv")
df

x = df[["Input"]].values
y = df[["Output"]].values
xt,xst,yt,yst = train_test_split(x,y,test_size=0.2,random_state=42)

scale1 = MinMaxScaler()
scale2=MinMaxScaler()
xt = scale1.fit_transform(xt)
xst = scale2.fit_transform(xst)


xt = torch.FloatTensor(xt)
xst = torch.FloatTensor(xst)
yt = torch.FloatTensor(yt)
yst = torch.FloatTensor(yst)

class neuralnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1,16),
            nn.ReLU(), 
            nn.Linear(16,8), 
            nn.ReLU(), 
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)

# Initialize the Model, Loss Function, and Optimizer
model = neuralnet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

# Train the model
epochs = 1000
losses=[]
for i in range(epochs):
    optimizer.zero_grad()
    pred = model(xt)
    loss = criterion(pred, yt)
    loss.backward()
    optimizer.step()

    if i % 50 == 0:
        print(f"{i}/{epochs} Loss: {loss.item():.4f}")
        losses.append(loss.item())

# Tesing for new input
new = scale1.transform([[16]])
new = torch.FloatTensor(new)

pred = model(new)
print(pred.item())

# Evaluating loss for testing data
with torch.no_grad():
    pred=model(xst)
    loss_test=criterion(pred,yst)
    print(loss_test)

# Plot the loss curve

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

```

### Dataset Information

<img width="252" height="592" alt="image" src="https://github.com/user-attachments/assets/577e641e-d172-4c31-b749-c7893f56ae41" />

### OUTPUT

### Training Loss Vs Iteration Plot

<img width="992" height="815" alt="image" src="https://github.com/user-attachments/assets/a6e9a858-beaf-4d58-ba30-fcafd8a9f4d1" />


### New Sample Data Prediction

<img width="766" height="847" alt="image" src="https://github.com/user-attachments/assets/aea72f9d-f43e-41a2-9088-55b99a39c534" />

<img width="1100" height="481" alt="image" src="https://github.com/user-attachments/assets/30b23cc6-c90a-4c0a-823a-a97215fe64ff" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
