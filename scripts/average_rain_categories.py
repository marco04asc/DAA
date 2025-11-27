import pandas as pd
import numpy as np

df_train = pd.read_csv('data/training_data.csv', encoding='latin1')
df_test  = pd.read_csv('data/test_data.csv', encoding='latin1')

print("="*70)
print("AVERAGE_RAIN ANALYSIS")
print("="*70)

# Get all unique categories (excluding NaN)
train_cats = set(df_train['AVERAGE_RAIN'].dropna().unique())
test_cats = set(df_test['AVERAGE_RAIN'].dropna().unique())
all_cats = train_cats.union(test_cats)

print(f"\n📊 SUMMARY:")
print(f"  Train unique (excl. NaN): {len(train_cats)}")
print(f"  Test unique (excl. NaN):  {len(test_cats)}")
print(f"  Total unique categories:  {len(all_cats)}")

# Categories only in TRAIN
only_train = train_cats - test_cats
print(f"\n🔴 Categories ONLY in TRAIN ({len(only_train)}):")
for cat in sorted(only_train):
    count = df_train['AVERAGE_RAIN'].value_counts()[cat]
    print(f"  - '{cat}': {count} occurrences")

# Categories only in TEST
only_test = test_cats - train_cats
print(f"\n🔵 Categories ONLY in TEST ({len(only_test)}):")
if only_test:
    for cat in sorted(only_test):
        count = df_test['AVERAGE_RAIN'].value_counts()[cat]
        print(f"  - '{cat}': {count} occurrences")
else:
    print("  None (all test categories appear in train ✅)")

# Categories in BOTH
both = train_cats.intersection(test_cats)
print(f"\n✅ Categories in BOTH ({len(both)}):")
print(f"  {sorted(both)}")

# DISTRIBUTION COMPARISON
print("\n" + "="*70)
print("DISTRIBUTION COMPARISON (categories in BOTH)")
print("="*70)

# Calculate percentages (excluding NaN)
train_total = df_train['AVERAGE_RAIN'].notna().sum()
test_total = df_test['AVERAGE_RAIN'].notna().sum()

comparison = []
for cat in sorted(both):
    train_count = df_train['AVERAGE_RAIN'].value_counts().get(cat, 0)
    test_count = df_test['AVERAGE_RAIN'].value_counts().get(cat, 0)
    
    train_pct = (train_count / train_total) * 100
    test_pct = (test_count / test_total) * 100
    
    diff = abs(train_pct - test_pct)
    
    comparison.append({
        'Category': cat,
        'Train_Count': train_count,
        'Train_%': train_pct,
        'Test_Count': test_count,
        'Test_%': test_pct,
        'Diff_%': diff
    })

df_comp = pd.DataFrame(comparison).sort_values('Diff_%', ascending=False)

print(df_comp.to_string(index=False))

# Missing values comparison
print("\n" + "="*70)
print("MISSING VALUES (NaN)")
print("="*70)
train_nan = df_train['AVERAGE_RAIN'].isna().sum()
test_nan = df_test['AVERAGE_RAIN'].isna().sum()
train_nan_pct = (train_nan / len(df_train)) * 100
test_nan_pct = (test_nan / len(df_test)) * 100

print(f"Train NaN: {train_nan} ({train_nan_pct:.1f}%)")
print(f"Test NaN:  {test_nan} ({test_nan_pct:.1f}%)")
