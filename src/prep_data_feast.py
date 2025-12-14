import pandas as pd
import os
from pathlib import Path
from datetime import datetime,timedelta
from sklearn.preprocessing import LabelEncoder

def create_parquet_for_feast(input_csv_path: str, output_parquet_path: str):
    print(f"--- Preparing {input_csv_path} for Feast ---")
   
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {input_csv_path}")
        return

    # Remove nulls if any
    df = df.dropna().reset_index(drop=True)
    print("✅ Removed null values.")

    # Drop "sno" column
    df = df.drop(columns=["sno"], errors='ignore')
    print("✅ Dropped 'sno' column")

    # Add a unique ID for each transaction
    if 'patient_id' not in df.columns:
        df['patient_id'] = df.index
        print("✅ Added 'patient_id' column.")

    # Add event_timestamp column required by Feast
    if 'event_timestamp' not in df.columns:
        df['event_timestamp'] = datetime.now()

    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

    # Encode categorical variables if any
    categorical_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"✅ Encoded categorical column: {col}")

    # Ensure the output directory exists
    output_dir = Path(output_parquet_path).parent
    os.makedirs(output_dir, exist_ok=True)
   
    # Save the processed DataFrame as a Parquet file
    df.to_parquet(output_parquet_path)
   
    print(f"💾 Successfully saved Feast-ready data to: {output_parquet_path}")
    print("---------------------------------------------------\n")


if __name__ == "__main__":
    input_path = "raw_data/heart.csv"
   
    # Define output paths for the new Parquet files
    output_path = "data/heart.parquet"

    # Process both datasets
    create_parquet_for_feast(input_path, output_path)
