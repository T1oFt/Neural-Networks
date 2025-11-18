import pandas as pd
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

def get_data():
    from ucimlrepo import fetch_ucirepo 
    
    # fetch dataset 
    mushroom = fetch_ucirepo(id=73) 
    
    # data (as pandas dataframes) 
    X = mushroom.data.features 
    y = mushroom.data.targets 
    
    # metadata 
    print(mushroom.metadata) 
    
    # variable information 
    print(mushroom.variables) 
    return X, y


X, y = get_data()

X['stalk-root'] = X['stalk-root'].fillna('-')

# print(X.info())

# print(X.describe())

print(y.describe())

def describe_dataset(df):
    C = len(df.columns)
    L = len(df.index)
    CN = df.count()
    NP = ((L - CN) / L) * 100
    P = df.nunique()
    
    frame = pd.concat([CN, NP, P], axis=1)
    frame = frame.T
    frame.index = ['Количество', 'Процент пропусков', 'Мощность']
    return frame


def describe_dataset_2(df):
    frame = {}
    for col in df.columns:
        if df[col].dtype == 'O' or str(df[col].dtype).startswith('category'):
            frame[col] = [
                df[col].count(),
                (df.shape[0] - df[col].count()) / df.shape[0] * 100,
                df[col].nunique(),
                df[col].value_counts().index[0],
                df[col].value_counts().iloc[0]
            ]
        else:
            frame[col] = [
                df[col].count(),
                (df.shape[0] - df[col].count()) / df.shape[0] * 100,
                df[col].min(),
                df[col].quantile(0.25),
                df[col].mean(),
                df[col].median(),
                df[col].quantile(0.75),
                df[col].max(),
                df[col].std(),
                df[col].nunique(),
                df[col].quantile(0.75) - df[col].quantile(0.25)
            ]
    return pd.DataFrame(frame)


print(describe_dataset_2(X))

def phi_coefficient(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape != (2, 2):
        return np.nan  # phi предназначен для 2x2 таблиц
    
    n = confusion_matrix.values.sum()
    a = confusion_matrix.iloc[0, 0]
    b = confusion_matrix.iloc[0, 1]
    c = confusion_matrix.iloc[1, 0]
    d = confusion_matrix.iloc[1, 1]
    
    numerator = a * d - b * c
    denominator = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        return np.nan
    return numerator / denominator

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    r, k = confusion_matrix.shape
    
    # Используем phi для 2x2 таблиц
    if r == 2 and k == 2:
        return abs(phi_coefficient(x, y))
    
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return np.nan
    
    return np.sqrt(phi2corr / denom)

def categorical_corr_matrix(df):
    cols = df.columns
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            val = cramers_v(df[cols[i]], df[cols[j]])
            mat.iloc[i, j] = val
            mat.iloc[j, i] = val
    return mat

def plot_heatmap(corr_mat):
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=0, vmax=1, square=True,
                cbar_kws={"shrink": .8})
    plt.title('Корреляционная матрица категориальных признаков (Cramér\'s V / Phi)')
    plt.show()

# Пример использования:
# df_cat - DataFrame с категориальными признаками
corr_mat = categorical_corr_matrix(pd.concat([X, y], axis=1))
plot_heatmap(corr_mat)

total = len(X)
same = (X['stalk-surface-above-ring'] == X['stalk-surface-below-ring']).sum()

print(f"Всего: {total}")
print(f"Одинаковых: {same}")
print(f"Разных: {total-same}")
print(f"Процент: {same / total * 100:.2f}%")
print(f"Процент: {(total-same) / total * 100:.2f}%")


def unique_value_percentages(df):
    result = {}
    for col in df.columns:
        value_counts = df[col].value_counts(normalize=True, dropna=False) * 100
        result[col] = value_counts.round(2).to_dict()
    return result

print(unique_value_percentages(X))

print(X.describe())

X.info()

X = X.drop(['veil-type', 'gill-attachment', 'veil-color', 'stalk-color-below-ring'], axis=1)