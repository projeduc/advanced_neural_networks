import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

# =========================================================
# DATA
# =========================================================

# people fish a little fish
X = torch.tensor([
    [-2.,  1.],   # people
    [ 0.,  5.],   # fish
    [ 1.,  2.],   # a
    [-1., -1.],   # little
    [ -3.,  2.]    # fish
])

# Targets:
# [0,0,1,0] -> class 2
# [0,0,0,1] -> class 3
# [0,1,0,0] -> class 1
# [1,0,0,0] -> class 0
# [0,0,1,0] -> class 2

y = torch.tensor([2, 3, 1, 0, 2])

# Add batch dimension
# shape: (batch, seq_len, input_size)
X = X.unsqueeze(0)

# =========================================================
# MODEL
# =========================================================

class ElmanRNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=2,
            hidden_size=3,
            nonlinearity='relu',
            batch_first=True
        )

        self.fc = nn.Linear(3, 4)

    def forward(self, x):

        # h0 shape:
        # (num_layers, batch, hidden_size)
        h0 = torch.zeros(1, x.size(0), 3)

        out, hn = self.rnn(x, h0)

        # out shape:
        # (batch, seq_len, hidden_size)

        out = self.fc(out)

        # output shape:
        # (batch, seq_len, 4)

        return out


model = ElmanRNN()

# =========================================================
# TRAINING
# =========================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):

    optimizer.zero_grad()

    outputs = model(X)

    # reshape:
    # (1,5,4) -> (5,4)
    loss = criterion(outputs.view(-1, 4), y)

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}  Loss = {loss.item():.4f}")

# =========================================================
# PREDICTIONS
# =========================================================

with torch.no_grad():

    outputs = model(X)

    probs = torch.softmax(outputs, dim=-1)

    preds = outputs.argmax(dim=-1)

    print("\nPredictions:")
    print(preds)

    print("\nProbabilities:")
    print(probs)

# =========================================================
# PRINT WEIGHTS
# =========================================================

print("\n==============================")
print("RNN INPUT -> HIDDEN WEIGHTS")
print("==============================")
print(model.rnn.weight_ih_l0)

print("\n==============================")
print("RNN HIDDEN -> HIDDEN WEIGHTS")
print("==============================")
print(model.rnn.weight_hh_l0)

print("\n==============================")
print("RNN BIASES")
print("==============================")
print(model.rnn.bias_ih_l0)
print(model.rnn.bias_hh_l0)

print("\n==============================")
print("DECODER WEIGHTS")
print("==============================")
print(model.fc.weight)

print("\n==============================")
print("DECODER BIAS")
print("==============================")
print(model.fc.bias)
