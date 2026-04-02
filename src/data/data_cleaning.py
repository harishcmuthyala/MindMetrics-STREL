import pandas as pd
from datetime import datetime

def load_data(filepath):
    """Load raw STREL data"""
    return pd.read_csv(filepath)

def remove_missing_target(df):
    """Remove rows where NHR_Stress is empty"""
    return df[df['NHR_Stress'].notna()].copy()

def remove_missing_rows(df):
    """Remove rows with any missing values"""
    return df.dropna(axis=0, how='any')

def extract_time_features(df):
    """Extract hour and minute from time column"""
    df['Time'] = pd.to_datetime(df['Time'])
    
    time_pos = df.columns.get_loc('Time')

    df.insert(time_pos + 1, 'Hour', df['Time'].dt.hour)
    df.insert(time_pos + 2, 'Minute', df['Time'].dt.minute)
    return df

def remove_columns(df, columns):
    """Remove specified columns"""
    return df.drop(columns=columns, errors='ignore')

def preprocess_pipeline(input_path, output_path):
    """Complete preprocessing pipeline"""
    df = load_data(input_path)
    df = remove_missing_target(df)
    df = extract_time_features(df)
    df = remove_columns(df, ['Date', 'Time', 'subDC'])
    df = remove_missing_rows(df)
    df.to_excel(output_path, index=False)
    print(f"File saved to: {output_path}")

if __name__ == "__main__":
    input_path = "../../data/raw/STREL_raw.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../../data/processed/processed_{timestamp}.xlsx"
    preprocess_pipeline(input_path, output_path)