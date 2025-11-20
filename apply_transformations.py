import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Carregar dados de treino e teste
df_train = pd.read_csv('data/training_data.csv', encoding='latin1')
df_test = pd.read_csv('data/test_data.csv', encoding='latin1')

# 1. Remover colunas constantes

def remove_constant_columns(df):
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    return df.drop(columns=constant_cols), constant_cols

df_train, constant_cols = remove_constant_columns(df_train)
df_test = df_test.drop(columns=constant_cols)
print(f'Removed constant columns: {constant_cols}')

# 2. Preencher valores vazios nas features com valores específicos

df_train['AVERAGE_RAIN'] = df_train['AVERAGE_RAIN'].fillna('sem chuva')
df_test['AVERAGE_RAIN'] = df_test['AVERAGE_RAIN'].fillna('sem chuva')

df_train['AVERAGE_SPEED_DIFF'] = df_train['AVERAGE_SPEED_DIFF'].fillna('None_Existent')
# df_test does not have AVERAGE_SPEED_DIFF, do not fill

df_train['AVERAGE_CLOUDINESS'] = df_train['AVERAGE_CLOUDINESS'].fillna('sem observação')
df_test['AVERAGE_CLOUDINESS'] = df_test['AVERAGE_CLOUDINESS'].fillna('sem observação')

# 3. Encoding das features categóricas

# Ordinal encoding for LIGHT_CONDITION
light_order = ['DARK', 'LOW_LIGHT', 'LIGHT']
oe_light = OrdinalEncoder(categories=[light_order])
df_train['LUMINOSITY'] = oe_light.fit_transform(df_train[['LUMINOSITY']])
df_test['LUMINOSITY'] = oe_light.transform(df_test[['LUMINOSITY']])

# Ordinal encoding for AVERAGE_CLOUDINESS
sky_order = [
    'céu limpo', 'céu claro', 'céu pouco nublado', 'algumas nuvens',
    'nuvens dispersas', 'nuvens quebradas', 'nuvens quebrados',
    'tempo nublado', 'nublado', 'sem observação'
]
oe_sky = OrdinalEncoder(categories=[sky_order])
df_train['AVERAGE_CLOUDINESS'] = oe_sky.fit_transform(df_train[['AVERAGE_CLOUDINESS']])
df_test['AVERAGE_CLOUDINESS'] = oe_sky.transform(df_test[['AVERAGE_CLOUDINESS']])

# Ordinal encoding for AVERAGE_RAIN
rain_order = [
    'sem chuva', 'chuvisco fraco', 'aguaceiros', 'aguaceiros fracos',
    'chuva leve', 'chuva fraca', 'chuva moderada', 'chuva',
    'chuva forte', 'chuva de intensidade pesado', 'chuva de intensidade pesada',
    'trovoada com chuva leve', 'trovoada com chuva', 'chuvisco e chuva fraca', 'Other values'
]
oe_rain = OrdinalEncoder(categories=[rain_order])
df_train['AVERAGE_RAIN'] = oe_rain.fit_transform(df_train[['AVERAGE_RAIN']])
df_test['AVERAGE_RAIN'] = oe_rain.transform(df_test[['AVERAGE_RAIN']])

# Ordinal encoding for target (only in train)
target_order = ['None_Existent', 'Low', 'Medium', 'High', 'Very_High']
oe_target = OrdinalEncoder(categories=[target_order])
df_train['AVERAGE_SPEED_DIFF'] = oe_target.fit_transform(df_train[['AVERAGE_SPEED_DIFF']])

X_train_final = df_train.drop(columns=['AVERAGE_SPEED_DIFF'])
y_train = df_train['AVERAGE_SPEED_DIFF']
X_test_final = df_test.copy()

X_train_final.to_csv('data/training_data_transformed.csv', index=False)
X_test_final.to_csv('data/test_data_transformed.csv', index=False)
y_train.to_csv('data/y_train_transformed.csv', index=False, header=True)

print('✅ Dados transformados e salvos!')

# Salvar os dados transformados
X_train_final.to_csv('data/training_data_transformed.csv', index=False)
X_test_final.to_csv('data/test_data_transformed.csv', index=False)
if y_train is not None:
    pd.Series(y_train, name='AVERAGE_SPEED_DIFF').to_csv('data/y_train_transformed.csv', index=False)

print('✅ Dados transformados e salvos!')
