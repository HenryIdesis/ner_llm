import pandas as pd

df = pd.read_excel('bd_05052020_anonimizado.xlsx', header=None)
cols = df.iloc[1].tolist()
print(f'Total de colunas: {len(cols)}\n')
print('Todas as colunas:')
for i, col in enumerate(cols, 1):
    print(f'{i:2d}. {col}')

