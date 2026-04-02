import pandas as pd
import random

def load_processed_data(filepath):
    """Load processed data"""
    return pd.read_excel(filepath)

def person_based_split(df, n_train):
    """Split data by number of train participants"""
    participants = df['Participant'].unique()
    train_participants = participants[:n_train]
    test_participants = participants[n_train:]
    
    train_df = df[df['Participant'].isin(train_participants)]
    test_df = df[df['Participant'].isin(test_participants)]
    return train_df, test_df

def save_splits(train_df, test_df, output_dir):
    """Save train and test sets"""
    train_df.to_excel(f"{output_dir}/train.xlsx", index=False)
    test_df.to_excel(f"{output_dir}/test.xlsx", index=False)
    print(f"Train set: {len(train_df)} rows, {train_df['Participant'].nunique()} participants")
    print(f"Test set: {len(test_df)} rows, {test_df['Participant'].nunique()} participants")
    print(f"Files saved to {output_dir}")

def create_5_folds(df, output_dir, seed=42):
    """
    Create 5 participant-based folds for cross-validation.
    Each fold uses a different group of participants as test set.
    Files saved as fold1_train.xlsx, fold1_test.xlsx, etc.

    Args:
        df: Full processed dataframe (must contain 'Participant' column)
        output_dir: Directory to save fold files
        seed: Random seed for reproducibility (default=42)
    """
    participants = list(df['Participant'].unique())
    print(f"Total participants: {len(participants)}")

    random.seed(seed)
    random.shuffle(participants)

    # Split 24 participants into 5 groups [5, 5, 5, 5, 4]
    folds = [participants[i::5] for i in range(5)]

    for i, test_participants in enumerate(folds, 1):
        train_participants = [p for p in participants if p not in test_participants]

        train_df = df[df['Participant'].isin(train_participants)]
        test_df  = df[df['Participant'].isin(test_participants)]

        train_df.to_excel(f"{output_dir}/fold{i}_train.xlsx", index=False)
        test_df.to_excel(f"{output_dir}/fold{i}_test.xlsx", index=False)

        print(f"Fold {i}: train={train_df['Participant'].nunique()} participants "
              f"({len(train_df)} rows), test={test_df['Participant'].nunique()} "
              f"participants ({len(test_df)} rows)")

    print(f"\nAll 5 folds saved to {output_dir}")


if __name__ == "__main__":
    input_path = "../../data/processed/processed_20260219_000215.xlsx"
    output_dir = "../../data/processed"
    
    df = load_processed_data(input_path)
    
    print("1. Single train/test split")
    print("2. 5-fold cross-validation splits")
    choice = input("\nChoose option (1 or 2): ")
    
    if choice == "1":
        total_participants = df['Participant'].nunique()
        print(f"Total participants: {total_participants}")
        n_train = int(input("How many for training? "))
        n_test = total_participants - n_train
        print(f"Test will have: {n_test} participants")
        train_df, test_df = person_based_split(df, n_train)
        save_splits(train_df, test_df, output_dir)
    
    elif choice == "2":
        create_5_folds(df, output_dir)
