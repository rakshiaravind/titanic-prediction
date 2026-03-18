import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def load_and_preprocess(path="train.csv"):
    df = pd.read_csv(path)

    # Feature engineering
    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        ["Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare"
    )
    df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    df["AgeBand"] = pd.cut(df["Age"], 5, labels=False)
    df["FareBand"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])
    df["Title"] = le.fit_transform(df["Title"])

    features = ["Pclass", "Sex", "AgeBand", "FareBand", "Embarked",
                "FamilySize", "IsAlone", "Title", "SibSp", "Parch"]
    X = df[features]
    y = df["Survived"]
    return X, y, features

def train_and_save(model_path="models/titanic_model.pkl"):
    X, y, features = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Did Not Survive", "Survived"]))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": features, "accuracy": acc}, f)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save()
