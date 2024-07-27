import json
import numpy as np
import torch
import torch.nn as nn

from nltk_utlis import tokenize, stemming, bag_of_words
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet


with open('data.json', 'r') as f:
    intents = json.load(f)

all_words = []
classes = []
xy= []

# loop through each sentence in each intents inputs
for intent in intents["intents"]:
    c_name = intent["class"]
    classes.append(c_name)

    for input in intent["inputs"]:
        w = tokenize(input)
        all_words.extend(w)
        xy.append((w,c_name))

# stem and lower each word
stop_words = ["?", "!", ".", ","]
all_words = [stemming(w) for w in all_words if w not in stop_words]
# remove duplicates and sort
all_words = sorted(set(all_words))

# create training data
X_train = []
Y_train = []

for (input, class_name) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(input, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels.
    label = classes.index(class_name)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = torch.tensor(Y_train, dtype=torch.long)

# Hyper-parameters 
batch_size = 8
hidden_size = 8
output_size = len(classes)
input_size = len(X_train[0])
learning_rate=0.001
n_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(n_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        output = model(words)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    if (epoch+1) % 100 == 0:
        print(f'epoch {epoch + 1}/{n_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "classes": classes
}

FILE = "data.pth"
torch.save(data, FILE)

print("training complete. file saved")