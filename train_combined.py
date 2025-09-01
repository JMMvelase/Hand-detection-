import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def load_kaggle_dataset(kaggle_path):
    """Load and preprocess the Kaggle sign language dataset."""
    try:
        # Load Kaggle dataset
        kaggle_df = pd.read_csv(kaggle_path)
        
        # Assuming Kaggle dataset has columns for landmarks x0-x20, y0-y20
        # Adjust column names if needed
        landmark_cols = ([f'x{i}' for i in range(21)] + 
                       [f'y{i}' for i in range(21)])
        
        # Ensure all required columns exist
        missing_cols = [col for col in landmark_cols if col not in kaggle_df.columns]
        if missing_cols:
            print(f"Warning: Missing columns in Kaggle dataset: {missing_cols}")
            return None
            
        # Select only the landmarks and label columns
        kaggle_df = kaggle_df[landmark_cols + ['label']]
        
        return kaggle_df
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}")
        return None

def load_sasl_dataset(sasl_path):
    """Load and preprocess the SASL dataset."""
    try:
        # Load SASL dataset
        sasl_df = pd.read_csv(sasl_path)
        
        # Select only the landmarks and label columns
        landmark_cols = ([f'x{i}' for i in range(21)] + 
                       [f'y{i}' for i in range(21)])
        sasl_df = sasl_df[landmark_cols + ['label']]
        
        return sasl_df
    except Exception as e:
        print(f"Error loading SASL dataset: {e}")
        return None

def combine_datasets(kaggle_df, sasl_df):
    """Combine Kaggle and SASL datasets."""
    if kaggle_df is None and sasl_df is None:
        raise ValueError("Both datasets failed to load!")
    
    # Initialize combined_df with the first non-None dataset
    if kaggle_df is not None:
        combined_df = kaggle_df.copy()
        if sasl_df is not None:
            combined_df = pd.concat([combined_df, sasl_df], ignore_index=True)
    else:
        combined_df = sasl_df.copy()
    
    return combined_df

def train_model(df, model_path="sasl_model.pkl"):
    """Train the model using the combined dataset."""
    # Prepare features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1  # Use all CPU cores
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nModel Performance:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    return model

def main():
    # File paths
    KAGGLE_PATH = "path_to_kaggle_dataset.csv"  # Update this
    SASL_PATH = "sasl_dataset/sasl_landmarks.csv"
    MODEL_PATH = "sasl_model.pkl"
    
    # Load datasets
    print("Loading Kaggle dataset...")
    kaggle_df = load_kaggle_dataset(KAGGLE_PATH)
    
    print("Loading SASL dataset...")
    sasl_df = load_sasl_dataset(SASL_PATH)
    
    # Combine datasets
    print("Combining datasets...")
    combined_df = combine_datasets(kaggle_df, sasl_df)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(combined_df)}")
    print("\nSamples per letter:")
    print(combined_df['label'].value_counts())
    
    # Train and save model
    model = train_model(combined_df, MODEL_PATH)

if __name__ == "__main__":
    main()
