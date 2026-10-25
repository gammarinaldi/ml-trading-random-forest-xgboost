import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def merge_eurusd_data():
    """
    Merge EUR/USD 1-minute data from 2000-2024
    """
    
    # Configuration
    # Source directory of the DAT_ASCII_EURUSD_M1_<YEAR>.csv archives.
    # Override with the EURUSD_M1_DIR environment variable.
    data_directory = os.environ.get(
        "EURUSD_M1_DIR",
        os.path.join("data", "eurusd"),
    )
    output_file = "EURUSD_M1_2000_2024_merged.csv"
    
    # Define column headers (semicolon-separated format)
    headers = [
        'DateTime',
        'Open', 
        'High',
        'Low',
        'Close',
        'Volume'
    ]
    
    print("EUR/USD 1-Minute Data Merger (2000-2024)")
    print("=" * 50)
    print(f"Source Directory: {data_directory}")
    print(f"Output File: {output_file}")
    print()
    
    # Check if directory exists
    if not os.path.exists(data_directory):
        print(f"Error: Directory not found: {data_directory}")
        return
    
    # Generate file list for years 2000-2024
    file_list = []
    missing_years = []
    
    for year in range(2000, 2025):  # 2000 to 2024 inclusive
        filename = f"DAT_ASCII_EURUSD_M1_{year}.csv"
        filepath = os.path.join(data_directory, filename)
        
        if os.path.exists(filepath):
            file_list.append((year, filepath))
            print(f"Found: {filename}")
        else:
            missing_years.append(year)
            print(f"Missing: {filename}")
    
    if missing_years:
        print(f"\nMissing {len(missing_years)} files for years: {missing_years}")
        response = input("Continue with available files? (y/n): ").lower().strip()
        if response != 'y':
            print("Operation cancelled.")
            return
    
    if not file_list:
        print("No data files found!")
        return
    
    print(f"\nProcessing {len(file_list)} files...")
    print("=" * 50)
    
    # Store all dataframes
    all_dataframes = []
    total_records = 0
    
    # Process each file
    for year, filepath in tqdm(file_list, desc="Loading files", unit="file"):
        try:
            # Read CSV with semicolon separator, no headers
            df = pd.read_csv(filepath, sep=';', header=None, names=headers)
            
            # Convert datetime
            df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H%M%S', errors='coerce')
            
            # Convert price columns to numeric
            price_cols = ['Open', 'High', 'Low', 'Close']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert volume to numeric
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            
            # Remove rows with invalid dates
            df = df.dropna(subset=['DateTime'])
            
            # Add year column for verification
            df['Year'] = year
            
            records_count = len(df)
            total_records += records_count
            
            all_dataframes.append(df)
            
            print(f"{year}: {records_count:,} records loaded")
            
        except Exception as e:
            print(f"Error loading {year}: {str(e)}")
            continue
    
    if not all_dataframes:
        print("No data could be loaded!")
        return
    
    print(f"\nMerging {len(all_dataframes)} dataframes...")
    
    # Combine all dataframes
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Initial merged records: {len(merged_df):,}")
    
    # Sort by datetime
    print("Sorting by datetime...")
    merged_df = merged_df.sort_values('DateTime').reset_index(drop=True)
    
    # Remove duplicates (if any)
    print("Removing duplicates...")
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=['DateTime']).reset_index(drop=True)
    duplicates_removed = initial_count - len(merged_df)
    
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed:,} duplicate records")
    
    # Data quality checks
    print("\nData Quality Report:")
    print("=" * 30)
    
    # Check for missing values
    missing_data = merged_df.isnull().sum()
    for col, missing in missing_data.items():
        if missing > 0:
            print(f"{col}: {missing:,} missing values")
    
    # Date range
    min_date = merged_df['DateTime'].min()
    max_date = merged_df['DateTime'].max()
    print(f"Date Range: {min_date} to {max_date}")
    
    # Records per year
    print(f"Records by Year:")
    year_counts = merged_df['Year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"    {year}: {count:,} records")
    
    # Price statistics
    print(f"Price Statistics:")
    print(f"    Min Close: {merged_df['Close'].min():.5f}")
    print(f"    Max Close: {merged_df['Close'].max():.5f}")
    print(f"    Avg Close: {merged_df['Close'].mean():.5f}")
    
    # Remove the temporary Year column before saving
    merged_df = merged_df.drop('Year', axis=1)
    
    # Save merged data
    print(f"\nSaving merged data to {output_file}...")
    
    try:
        # Save as CSV with proper headers
        merged_df.to_csv(output_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        
        print(f"Successfully saved!")
        print(f"File: {output_file}")
        print(f"Records: {len(merged_df):,}")
        print(f"Size: {file_size:.1f} MB")
        
        # Create a sample file for inspection
        sample_file = "EURUSD_sample_1000.csv"
        sample_df = merged_df.head(1000)
        sample_df.to_csv(sample_file, index=False, date_format='%Y-%m-%d %H:%M:%S')
        print(f"Sample: {sample_file} (first 1,000 records)")
        
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return
    
    print(f"\nEUR/USD Data Merge Complete!")
    print("=" * 50)
    print(f"Total Records: {len(merged_df):,}")
    print(f"Time Span: {max_date - min_date}")
    print(f"Output File: {output_file}")
    
    # Show final data preview
    print(f"\nData Preview (First 5 records):")
    print(merged_df.head().to_string(index=False))
    
    print(f"\nData Preview (Last 5 records):")
    print(merged_df.tail().to_string(index=False))

def verify_merged_data(filename="EURUSD_M1_2000_2024_merged.csv"):
    """
    Quick verification of the merged data
    """
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return
    
    print(f"\nVerifying merged data: {filename}")
    print("=" * 40)
    
    try:
        # Load just the first few rows to check structure
        df_sample = pd.read_csv(filename, nrows=10)
        
        print(f"Columns: {list(df_sample.columns)}")
        print(f"Sample Data:")
        print(df_sample.to_string(index=False))
        
        # Get file info
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        
        # Count total lines (approximate record count)
        with open(filename, 'r') as f:
            line_count = sum(1 for line in f) - 1  # subtract header
        
        print(f"\nFile Statistics:")
        print(f"Size: {file_size:.1f} MB")
        print(f"Estimated Records: {line_count:,}")
        
    except Exception as e:
        print(f"Error verifying file: {str(e)}")

if __name__ == "__main__":
    # Run the merger
    merge_eurusd_data()
    
    # Verify the output
    verify_merged_data()
    
    print(f"\nUsage Tips:")
    print(f"  • Use the merged file for backtesting")
    print(f"  • Load with: pd.read_csv('EURUSD_M1_2000_2024_merged.csv', parse_dates=['DateTime'])")
    print(f"  • Check the sample file first for data format verification") 