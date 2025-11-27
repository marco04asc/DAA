import pandas as pd

# Load Data
print("Loading raw data...")
df_train = pd.read_csv('data/training_data.csv', encoding='latin1')
df_test = pd.read_csv('data/test_data.csv', encoding='latin1')

# ============== 1. DATE & ID TRANSFORMATION (Crucial Step) ==============
# Since record_date is the ID, we need to extract info BEFORE turning it into a simple number
print("Processing Dates...")

def process_dates(df):
    # Convert to datetime object
    df['record_date'] = pd.to_datetime(df['record_date'])
    
    # Extract universal traffic features
    df['Hour'] = df['record_date'].dt.hour
    df['DayOfWeek'] = df['record_date'].dt.dayofweek # 0=Mon, 6=Sun
    df['Month'] = df['record_date'].dt.month
    
    # Note: We do NOT drop 'record_date' yet if we want to verify order, 
    # but we will replace it with an integer ID for the final file.
    return df

df_train = process_dates(df_train)
df_test = process_dates(df_test)

# Create a clean integer ID (useful for submissions/tracking)
# Assuming the data is already in order. If not, sort by date first!
# df_train = df_train.sort_values('record_date') # Uncomment if needed
df_train['RowId'] = range(1, len(df_train) + 1)
df_test['RowId'] = range(1, len(df_test) + 1)

# Now drop the original string date column (we extracted the value already)
df_train = df_train.drop(columns=['record_date'])
df_test = df_test.drop(columns=['record_date'])

# ============== 2. REMOVE CONSTANT COLUMNS ==============
constant_cols = [col for col in df_train.columns if df_train[col].nunique() == 1]
df_train = df_train.drop(columns=constant_cols)
df_test = df_test.drop(columns=constant_cols)
print(f"Removed constant columns: {constant_cols}")

# ============== 3. IMPUTATION (Semantic) ==============
# Rain: NaN -> 'sem chuva'
df_train['AVERAGE_RAIN'] = df_train['AVERAGE_RAIN'].fillna('sem chuva')
df_test['AVERAGE_RAIN'] = df_test['AVERAGE_RAIN'].fillna('sem chuva')

# Cloudiness: NaN -> 'sem observação' (Keep informative missingness)
df_train['AVERAGE_CLOUDINESS'] = df_train['AVERAGE_CLOUDINESS'].fillna('sem observação')
df_test['AVERAGE_CLOUDINESS'] = df_test['AVERAGE_CLOUDINESS'].fillna('sem observação')

# ============== 4. TARGET HANDLING (Train Only) ==============
# NaN Target -> 'None_Existent' class
df_train['AVERAGE_SPEED_DIFF'] = df_train['AVERAGE_SPEED_DIFF'].fillna('None_Existent')

# ============== 5. SAVE ==============
# Move RowId to the front for cleanliness (optional)
cols_train = ['RowId'] + [c for c in df_train.columns if c != 'RowId']
cols_test = ['RowId'] + [c for c in df_test.columns if c != 'RowId']

df_train = df_train[cols_train]
df_test = df_test[cols_test]

df_train.to_csv('data/training_data_universal.csv', index=False)
df_test.to_csv('data/test_data_universal.csv', index=False)

print("Universal preprocessing (Dates + IDs + Imputation) complete.")
print("Files saved: 'data/training_data_universal.csv', 'data/test_data_universal.csv'")
