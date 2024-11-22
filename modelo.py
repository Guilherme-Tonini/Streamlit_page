from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # O construtor não precisa de parâmetros
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        # Renomeia as colunas e remove a primeira linha (que contém o cabeçalho antigo)
        df.columns = ['Data', 'Preço']
        df = df.drop(0)
        # Adiciona o ponto após o segundo número e transforma para float
        df['Preço'] = df['Preço'].apply(lambda x: float(str(x)[:-2] + '.' + str(x)[-2:]))
        # Transforma a coluna data para o tipo datetime
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
        # Define a coluna data como o índice do DataFrame
        df.set_index('Data', inplace=True)
        return df
    

class TimeSeriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass  # O construtor não precisa de parâmetros

    def fit(self, df):
        return self
    
    def transform(self, df):
        # Define a frequência do DataFrame para diário e preenche valores NaN com a média
        df = df.asfreq('D')
        media = df.mean()
        df = df.fillna(media)

        # Mantém apenas os últimos 6 meses de dados
        data_mais_recente = df.index.max()
        tempo = data_mais_recente - pd.DateOffset(months=6)
        df = df.loc[tempo:data_mais_recente]

        # Salva o último valor original antes da transformação para futura reversão
        ultimo_valor_original = df['Preço'].iloc[-1]

        # Aplica logaritmo aos dados e, em seguida, calcula a diferença
        df = np.log(df)
        df = df.diff(1).dropna()  # Remove NaN resultantes da diferenciação

        # Prepara o DataFrame para uso no modelo de previsão
        df = df.reset_index()
        df.rename(columns={'Data': 'ds', 'Preço': 'y'}, inplace=True)
        df['unique_id'] = 'Brent'
        df = df[['unique_id', 'ds', 'y']]

        # Divide os dados entre treinamento e previsão
        df_valores = df[:-14]  # Exclui os últimos 14 dias para previsão

        # Configura e treina o modelo de previsão com AutoARIMA
        modelo = StatsForecast(models=[AutoARIMA(season_length=7, stationary=True, seasonal=True)], freq='D', n_jobs=-1)
        modelo.fit(df_valores)
        
        # Faz a previsão para os próximos 19 dias e mantém os últimos 6
        previsao = modelo.predict(h=19, level=[90]).iloc[-6:]

        # Processa o DataFrame de previsão e ajusta as colunas
        previsao.rename(columns={'AutoARIMA': 'Preço', 'ds': 'Data'}, inplace=True)
        previsao.set_index('Data', inplace=True)
        previsao.iloc[0] = 0  # Ajusta o primeiro valor da previsão para 0 (base da cumulativa)

        # Reverte a transformação diferencial e exponencial para obter valores reais
        previsao['Preço'] = previsao['Preço'].cumsum() + np.log(ultimo_valor_original)
        previsao['Preço'] = np.exp(previsao['Preço'])

        # Transformação das colunas de intervalo de confiança para o nível original
        previsao['AutoARIMA-lo-90'] = previsao['AutoARIMA-lo-90'].cumsum() + np.log(ultimo_valor_original)
        previsao['AutoARIMA-lo-90'] = np.exp(previsao['AutoARIMA-lo-90'])

        previsao['AutoARIMA-hi-90'] = previsao['AutoARIMA-hi-90'].cumsum() + np.log(ultimo_valor_original)
        previsao['AutoARIMA-hi-90'] = np.exp(previsao['AutoARIMA-hi-90'])

        previsao.rename(columns={'AutoARIMA-lo-90': 'Limite Inferior (90%)', 'AutoARIMA-hi-90': 'Limite Superior (90%)'}, inplace=True)

        # Exclui o primeiro dia da previsão e retorna o DataFrame final
        df = previsao.iloc[1:]
        return df


