# 🚢 Titanic Survival Predictor

A Streamlit app that predicts passenger survival on the Titanic using a **Random Forest** classifier.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**1. Add the dataset**
Place the Titanic `train.csv` into the `data/` folder.
Download from: https://www.kaggle.com/c/titanic/data

**2. Train the model**
```bash
python train_model.py
```
This preprocesses the data, trains a Random Forest (200 trees), prints accuracy, and saves the model to `models/titanic_model.pkl`.

**3. Run the app**
```bash
streamlit run app.py
```

## Features

- Passenger input: class, sex, age, fare, port, family size, title
- Survival prediction with probability gauge
- Feature importance visualization
- Historical survival rate context chart
- Model accuracy display

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Max Depth | 6 |
| Test Accuracy | ~83% |
| Features | 10 engineered features |

## Project Structure

```
titanic-survival-app/
├── data/
│   └── train.csv          # Titanic dataset (add manually)
├── models/
│   └── titanic_model.pkl  # Saved after training
├── app.py                 # Streamlit UI
├── train_model.py         # Training script
└── requirements.txt
```
