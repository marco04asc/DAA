import pandas as pd

df_train = pd.read_csv('data/training_data.csv', encoding='latin1')

# 1. Fill target just for this check (so we can see the distribution)
df_train['AVERAGE_SPEED_DIFF'] = df_train['AVERAGE_SPEED_DIFF'].fillna('None_Existent')

# 2. Create a flag: is cloudiness missing?
df_train['cloud_missing'] = df_train['AVERAGE_CLOUDINESS'].isna()

print("Target Distribution when Cloudiness is MISSING:")
print(df_train[df_train['cloud_missing']]['AVERAGE_SPEED_DIFF'].value_counts(normalize=True).sort_index())

print("\nTarget Distribution when Cloudiness is PRESENT:")
print(df_train[~df_train['cloud_missing']]['AVERAGE_SPEED_DIFF'].value_counts(normalize=True).sort_index())
