import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# 1. Load the UNIVERSAL file (which now has Hour, Day, Month)
df = pd.read_csv('data/training_data_universal.csv')

# 2. Encode Target Temporarily (Just for correlation check)
target_map = {'None_Existent': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
df['Target_Num'] = df['AVERAGE_SPEED_DIFF'].map(target_map)

# 3. Calculate Spearman Correlations
# We check: Hour, DayOfWeek, Month vs Target
date_features = ['Hour', 'DayOfWeek', 'Month']
correlations = df[date_features + ['Target_Num']].corr(method='spearman')['Target_Num'].drop('Target_Num')

print("\nCorrelation with Traffic Speed Difference:")
print(correlations.sort_values(ascending=False))

# 4. Visualization (The best way to see patterns)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Hour vs Traffic
sns.boxplot(x='Hour', y='Target_Num', data=df, ax=axes[0], palette='viridis')
axes[0].set_title(f"Hour of Day (Corr: {correlations['Hour']:.2f})")
axes[0].set_ylabel("Traffic Impact (0=None, 4=Very High)")
axes[0].set_xlabel("Hour (0-23)")

# Plot 2: Day of Week vs Traffic
sns.boxplot(x='DayOfWeek', y='Target_Num', data=df, ax=axes[1], palette='coolwarm')
axes[1].set_title(f"Day of Week (Corr: {correlations['DayOfWeek']:.2f})")
axes[1].set_xlabel("Day (0=Mon, 6=Sun)")
axes[1].set_ylabel("")

# Plot 3: Month vs Traffic
sns.boxplot(x='Month', y='Target_Num', data=df, ax=axes[2], palette='magma')
axes[2].set_title(f"Month (Corr: {correlations['Month']:.2f})")
axes[2].set_xlabel("Month (1-12)")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()
