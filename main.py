import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from scarf.loss import NTXent
from scarf.model import SCARF

from example.dataset import ExampleDataset
from example.utils import dataset_embeddings, fix_seed, train_epoch

seed = 1234
fix_seed(seed)


# preprocess your data and create your pytorch dataset
# train_ds = ...
data = datasets.load_breast_cancer(as_frame=True)
data, target = data["data"], data["target"]
train_data, test_data, train_target, test_target = train_test_split(
    data,
    target,
    test_size=0.2,
    stratify=target,
    random_state=seed
)

# preprocess
constant_cols = [c for c in train_data.columns if train_data[c].nunique() == 1]
train_data.drop(columns=constant_cols, inplace=True)
test_data.drop(columns=constant_cols, inplace=True)

scaler = StandardScaler()
train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

# to torch dataset
train_ds = ExampleDataset(
    train_data.to_numpy(),
    train_target.to_numpy(),
    columns=train_data.columns
)
test_ds = ExampleDataset(
    test_data.to_numpy(),
    test_data.to_numpy(),
    columns=test_data.columns
)

print(f"Train set: {train_ds.shape}")
print(f"Test set: {test_ds.shape}")
train_ds.to_dataframe().head()

# train the model
batch_size = 128
epochs = 5000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

model = SCARF(
    input_dim=train_ds.shape[1],
    emb_dim=16,
    features_low=train_ds.features_low,
    features_high=train_ds.features_high,
    corruption_rate=0.6,
    dropout=0.1
).to(device)

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
ntxent_loss = NTXent()

loss_history = []

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    for x in train_loader:
        x = x.to(device)
        emb_anchor, emb_positive = model(x)

        loss = criterion(emb_anchor, emb_positive)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()

    loss_history.append(epoch_loss)

    if epoch % 100 == 0:
      print(f"epoch {epoch}/{epochs} - loss: {loss_history[-1]:.4f}")