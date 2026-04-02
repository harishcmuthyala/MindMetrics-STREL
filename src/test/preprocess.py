import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """Load raw STREL data"""
    return pd.read_csv(filepath)

def remove_missing_target(df):
    """Remove rows where NHR_Stress is empty"""
    return df[df['NHR_Stress'].notna()].copy()

def drop_missing_physiological(df):
    """Drop rows with missing values in key physiological columns"""
    columns_to_check = ['RR', 'Cadence', 'PNSindex', 'SNSindex', 'Stressindex']
    return df.dropna(subset=columns_to_check).copy()

def merge_workdays(df):
    """Merge WD1 and WD2 into WD"""
    df['Day'] = df['Day'].replace({'WD1': 'WD', 'WD2': 'WD'})
    return df

def sort_data(df):
    """Sort by participant and time"""
    return df.sort_values(['Participant', 'Date', 'Time']).reset_index(drop=True)

def extract_temporal(df):
    """Extract Hour and Minute from Time"""
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute
    return df

def encode_categoricals(df):
    """Encode categorical features"""
    categoricals = ['Day', 'Period', 'Activity4', 'Profession', 'Gender']
    for col in categoricals:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    return df

def encode_target(df):
    """Encode target variable: NS=0, S=1"""
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['NHR_Stress'])
    return df

def select_features(df):
    """Select 26 features for training"""
    features = [
        'Day_encoded', 'Period_encoded', 'Activity4_encoded', 
        'Profession_encoded', 'Gender_encoded',
        'Age', 'BMI',
        'HR', 'RR', 'PNSindex', 'SNSindex', 'HR_Baseline',
        'Cadence', 'Speed',
        'Sleep_Time', 'PA_Percent', 'PA_Intensity',
        'N_MD', 'N_PD', 'N_P',
        'Extraversion', 'Agreeableness', 'Neuroticism', 'Openness',
        'Hour', 'Minute'
    ]
    X = df[features]
    y = df['Target']
    participant = df['Participant']
    return X, y, participant

def save_processed_data(X, y, participant, output_path):
    """Save processed data to Excel"""
    df_final = X.copy()
    df_final['Target'] = y.values
    df_final['Participant'] = participant.values
    cols = ['Participant'] + [col for col in df_final.columns if col not in ['Participant', 'Target']] + ['Target']
    df_final = df_final[cols]
    df_final.to_excel(output_path, index=False)

def preprocess_pipeline(input_path, output_path):
    """Complete preprocessing pipeline"""
    df = load_data(input_path)
    df = remove_missing_target(df)
    df = drop_missing_physiological(df)
    df = merge_workdays(df)
    df = sort_data(df)
    df = extract_temporal(df)
    df = encode_categoricals(df)
    df = encode_target(df)
    X, y, participant = select_features(df)
    save_processed_data(X, y, participant, output_path)
    return X, y, participant

if __name__ == "__main__":
    input_path = "../../data/raw/STREL_raw.csv"
    output_path = "../../data/processed/processed_data.xlsx"
    X, y, participant = preprocess_pipeline(input_path, output_path)
