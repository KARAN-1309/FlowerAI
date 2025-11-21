import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
# Adjust this path if your CSV is located elsewhere
DATASET_PATH = os.path.join('Dataset', 'Iris', 'iris.csv')
MODEL_PATH = "iris_tabular_model.pkl"
ENCODER_PATH = "iris_label_encoder.pkl"

def train_model():
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        print(f"Error: File not found at {DATASET_PATH}")
        print("Please ensure 'Dataset/Iris/iris.csv' exists.")
        return

    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print("Dataset columns found:", df.columns.tolist())

    # 2. Preprocessing
    # We assume the standard Iris dataset structure.
    # We need to find the target column (the species name).
    target_col = None
    for col in df.columns:
        if df[col].dtype == 'object':
            target_col = col
            break
    
    if target_col is None:
        # Fallback: Assume last column is target if no strings found
        target_col = df.columns[-1]

    print(f"Target column identified as: '{target_col}'")

    # Separate features (measurements) and target (species)
    X = df.drop(columns=[target_col])
    
    # Remove 'Id' column if it exists (common in Kaggle versions of this dataset)
    if 'Id' in X.columns:
        X = X.drop(columns=['Id'])
    
    y = df[target_col]

    print(f"Features being used: {X.columns.tolist()}")

    # Encode target labels (e.g., 'Iris-setosa' becomes 0)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 3. Train Model
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # 4. Evaluate
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model Accuracy: {acc * 100:.2f}%")

    # 5. Save Artifacts
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label Encoder saved to {ENCODER_PATH}")

if __name__ == "__main__":
    train_model()