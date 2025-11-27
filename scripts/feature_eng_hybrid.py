import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# ============== 1. LOAD DATA ==============
print("Loading universal base data...")
df_train = pd.read_csv('data/training_data_universal.csv', encoding='latin1')
df_test = pd.read_csv('data/test_data_universal.csv', encoding='latin1')

# ============== 2. HYBRID ENCODING ==============

# --- A) ORDINAL: LUMINOSITY (Physics: Dark < Low < Light) ---
light_order = ['DARK', 'LOW_LIGHT', 'LIGHT']
oe_light = OrdinalEncoder(categories=[light_order], handle_unknown='use_encoded_value', unknown_value=-1)
df_train['LUMINOSITY'] = oe_light.fit_transform(df_train[['LUMINOSITY']])
df_test['LUMINOSITY'] = oe_light.transform(df_test[['LUMINOSITY']])

# --- B) ORDINAL: AVERAGE_RAIN (Physics: None < Light < Moderate < Heavy) ---
# Explicit map based on your data analysis
rain_map_simplified = {
    # Level 0: None
    'sem chuva': 0,
    
    # Level 1: Light
    'chuvisco fraco': 1, 'chuvisco e chuva fraca': 1,
    'chuva leve': 1, 'chuva fraca': 1, 'aguaceiros fracos': 1,
    
    # Level 2: Moderate
    'chuva moderada': 2, 'aguaceiros': 2, 'chuva': 2, 
    'trovoada com chuva leve': 2,
    
    # Level 3: Heavy
    'chuva forte': 3, 'chuva de intensidade pesado': 3, 
    'chuva de intensidade pesada': 3, 'trovoada com chuva': 3
}

df_train['AVERAGE_RAIN'] = df_train['AVERAGE_RAIN'].map(rain_map_simplified).fillna(0).astype(int)
df_test['AVERAGE_RAIN'] = df_test['AVERAGE_RAIN'].map(rain_map_simplified).fillna(0).astype(int)

# --- C) ONE-HOT: AVERAGE_CLOUDINESS (No clear order) ---
# This creates columns like 'CLOUD_céu limpo', 'CLOUD_sem observação', etc.
df_train = pd.get_dummies(df_train, columns=['AVERAGE_CLOUDINESS'], prefix='CLOUD')
df_test = pd.get_dummies(df_test, columns=['AVERAGE_CLOUDINESS'], prefix='CLOUD')

# --- D) ORDINAL: TARGET (Train Only) ---
target_order = ['None_Existent', 'Low', 'Medium', 'High', 'Very_High']
oe_target = OrdinalEncoder(categories=[target_order])
df_train['AVERAGE_SPEED_DIFF'] = oe_target.fit_transform(df_train[['AVERAGE_SPEED_DIFF']])

# ============== 3. ALIGNMENT & SAVING ==============

# Separate Features and Target
X_train = df_train.drop(columns=['AVERAGE_SPEED_DIFF'])
y_train = df_train['AVERAGE_SPEED_DIFF']
X_test = df_test.copy()

# CRITICAL STEP: Align Test columns to match Train exactly
# 1. Adds missing columns to Test (filled with 0)
# 2. Removes extra columns from Test (that Train doesn't have)
# 3. Reorders columns to match Train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save
print("Saving hybrid encoded files...")
X_train.to_csv('data/training_data_hybrid.csv', index=False)
X_test.to_csv('data/test_data_hybrid.csv', index=False)
y_train.to_csv('data/y_train_hybrid.csv', index=False, header=True)

print('Success! Data transformed (Hybrid Encoding) and saved.')
print(f'Features: {X_train.shape[1]} | Samples: {X_train.shape[0]}')
