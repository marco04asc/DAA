import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder

# 1. Load your clean base data
df = pd.read_csv('data/training_data_universal.csv')

# 2. Encode Target (Order matters for correlation!)
target_map = {'None_Existent': 0, 'Low': 1, 'Medium': 2, 'High': 3, 'Very_High': 4}
df['target_enc'] = df['AVERAGE_SPEED_DIFF'].map(target_map)

# 3. Apply your Manual Rain Map (The "Cool" Option)
rain_map = {
    'sem chuva': 0,
    'chuvisco fraco': 1, 'chuvisco e chuva fraca': 1,
    'chuva leve': 2, 'chuva fraca': 2, 'aguaceiros fracos': 2,
    'chuva moderada': 3, 'aguaceiros': 3, 'chuva': 3, 'trovoada com chuva leve': 3,
    'chuva forte': 4, 'chuva de intensidade pesado': 4, 'chuva de intensidade pesada': 4, 'trovoada com chuva': 4
}
df['rain_ordinal'] = df['AVERAGE_RAIN'].map(rain_map).fillna(0)

# 4. Compare Correlations
# Spearman is best because data is ordinal (ranks), not strictly linear
corr_ordinal = df[['rain_ordinal', 'target_enc']].corr(method='spearman').iloc[0,1]

print(f"Correlation (Rain Intensity vs Speed Diff): {corr_ordinal:.4f}")

# 5. Visualization check
plt.figure(figsize=(8, 5))
sns.boxplot(x='rain_ordinal', y='target_enc', data=df)
plt.title(f"Does Rain Intensity affect Traffic? (Corr: {corr_ordinal:.2f})")
plt.xlabel("Rain Intensity (0=None, 4=Heavy)")
plt.ylabel("Speed Diff (0=None, 4=Very High)")
plt.show()
