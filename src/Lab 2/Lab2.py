import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Zadanie 1
X, y = load_wine(return_X_y=True)
# Podział 80/20 z zachowaniem proporcji klas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Zadanie 2
# Tworzenie pipeline
sk_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, max_iter=200))
])

start_time = time.time()
# Trening
sk_pipeline.fit(X_train, y_train)
sk_time = time.time() - start_time

# Predykcja i ocena
sk_pred = sk_pipeline.predict(X_test)
sk_acc = accuracy_score(y_test, sk_pred)

# Zadanie 3
# 1. Definicja sieci
pt_model = nn.Sequential(
    nn.Linear(13, 16),
    nn.ReLU(),
    nn.Linear(16, 3)
)

# 2. Przygotowanie danych, skalowanie i konwersja na Tensory
scaler = StandardScaler()
X_train_pt = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.long)
X_test_pt = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_test_pt = torch.tensor(y_test, dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pt_model.parameters(), lr=0.01)

# 3. Pętla treningowa
start_time = time.time()
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(pt_model(X_train_pt), y_train_pt)
    loss.backward()
    optimizer.step()
pt_time = time.time() - start_time

# 4. Ewaluacja modelu na zbiorze testowym
with torch.no_grad():
    pt_pred = pt_model(X_test_pt).argmax(dim=1).numpy()
    pt_acc = accuracy_score(y_test, pt_pred)

# Zadanie 4
kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Uruchomienie walidacji krzyżowej
sk_scores = cross_val_score(sk_pipeline, X, y, cv=kf, scoring="accuracy")

# Zadanie 5
# 1. Przygotowanie całego zbioru jako tensory
X_tensor = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

pt_scores = []

# 2. Pętla przez 5 foldów (korzystamy z kf z Zadania 4)
for train_idx, val_idx in kf.split(X):

    # Tworzymy NOWY model dla każdego foldu aby wagi uczyły się od zera
    fold_model = nn.Sequential(
        nn.Linear(13, 16),
        nn.ReLU(),
        nn.Linear(16, 3)
    )
    optimizer = optim.Adam(fold_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Trening foldu
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(fold_model(X_tensor[train_idx]), y_tensor[train_idx])
        loss.backward()
        optimizer.step()

    # Ewaluacja na części walidacyjnej
    with torch.no_grad():
        pred = fold_model(X_tensor[val_idx]).argmax(dim=1).numpy()
        acc = accuracy_score(y_tensor[val_idx].numpy(), pred)
        pt_scores.append(acc)

# --- PODSUMOWANIE ---
pt_scores = np.array(pt_scores)
print(f"| scikit-learn | Acc: {sk_acc:.4f} | Czas: {sk_time:.4f}s | CV: {sk_scores.mean():.4f} ± {sk_scores.std():.4f} |")
print(f"| PyTorch      | Acc: {pt_acc:.4f} | Czas: {pt_time:.4f}s | CV: {pt_scores.mean():.4f} ± {pt_scores.std():.4f} |")