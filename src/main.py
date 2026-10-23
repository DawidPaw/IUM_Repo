import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Wczytanie konfiguracji z pliku .env
load_dotenv()

data_path = os.getenv("DATA_PATH", "data/wine.csv")
test_size = float(os.getenv("TEST_SIZE", 0.2))
random_seed = int(os.getenv("RANDOM_SEED", 42))

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

model = DecisionTreeClassifier(random_state=random_seed)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

with open("wyniki.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(f"Raport klasyfikacji:\n{report}\n")