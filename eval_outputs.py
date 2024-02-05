import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
import numpy as np

def main(results_csv_path):
    # Load the results.csv file with tab as separator
    df_results = pd.read_csv(results_csv_path, sep='\t')

    # Ensure 'ambig_status' column is in integer format
    df_results['ambig_status'] = df_results['ambig_status'].astype(int)
    
    # Check if 'binary' column exists in the DataFrame
    if 'binary' not in df_results:
        # If 'binary' column doesn't exist, create it based on 'response'
        df_results['binary'] = df_results['response'].apply(lambda response: np.nan if pd.isna(response) else (1 if 'yes' in response.lower() else (0 if 'no' in response.lower() else np.nan)))

    # Ensure 'binary' column is in integer format
    df_results['binary'] = df_results['binary'].astype(float)

    # Count the occurrences of each unique value in 'binary' column, including NaN
    binary_counts = df_results['binary'].value_counts(dropna=False)
    print("Counts in df_results['binary'] (including NaN):")
    print(binary_counts)

    # Drop rows with NaN values in 'binary' column
    df_results.dropna(subset=['binary'], inplace=True)

    # Calculate accuracy score
    accuracy = accuracy_score(df_results['ambig_status'], df_results['binary'])

    print(f"\nAccuracy Score: {accuracy:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy score from results CSV file.')
    parser.add_argument('results_csv_path', type=str, help='Path to the results CSV file')
    args = parser.parse_args()

    main(args.results_csv_path)

