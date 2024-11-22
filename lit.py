import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from modelo import DataFrameTransformer, TimeSeriesTransformer
import plotly.graph_objects as go
import plotly.express as px
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast

# URL do site com os dados da série do IPEADATA
url = 'http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'

# Enviar uma requisição GET para o site
response = requests.get(url)

# Verificar se a requisição foi bem-sucedida (status 200)
if response.status_code == 200:
    # Usar o BeautifulSoup para fazer o parsing do HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Encontrar a tabela no HTML (geralmente as tabelas estão em tags <table>)
    table = soup.find('table', {'class': 'dxgvTable'})

    # Ler a tabela diretamente para um DataFrame do pandas
    dff = pd.read_html(str(table))[0]

def analysis(df): #Cria um pipeline e aplica transformações ao DataFrame fornecido.

    # Criar o pipeline
    pipeline = Pipeline(steps=[
        ('dataframe_transformer', DataFrameTransformer()),  # Primeira transformação
    ])
    
    # Aplicar o pipeline ao DataFrame
    transformed_df = pipeline.fit_transform(df)
    
    return transformed_df

df = analysis(dff)

st.markdown("<h1 style='text-align: left; font-size: 6em;'>Tech Challenge</h1>", unsafe_allow_html=True)
st.markdown('<br><br><br>', unsafe_allow_html=True)
st.markdown('## Introdução', unsafe_allow_html=True)
st.markdown(''' <div style="text-align: justify;">
         Esta página corresponde a um projeto referente ao Tech Challenge IV
         do curso pós tech em análise de dados da instituição FIAP.<br>
         Este projeto tem o intuito de aprensentar um Dashboard interativo
         sobre a variação de preço (em dólares) do barril de petróleo bruto Brent produzido no Mar do Norte, 
         além de uma demonstração de adaptação de um modelo
         de machine learn em relação a sasonalidade dos dados com a previsão dos próximos dias. <br>
         Os valores serão importados diretamente do site da IPEA, Instituto de Pesquisa Econômica Aplicada, 
         uma fundação pública federal vinculada ao Ministério do Planejamento e Orçamento.
         O site é atualizado semanalmente com preços correspondentes aos dias úteis.
         </div>''', unsafe_allow_html=True)


st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('## Dashboard', unsafe_allow_html=True)

# Obter o ano mínimo e máximo do índice
min_year = df.index.year.min()
max_year = df.index.year.max()

# Barra de seleção de intervalo de anos
year_range = st.slider("Selecione o intervalo de anos:", min_value=min_year, max_value=max_year, value=(min_year, max_year))

# Filtrar o DataFrame pelo intervalo de anos selecionado
filtered_data = df[(df.index.year >= year_range[0]) & (df.index.year <= year_range[1])]

# Cálculo das médias móveis
filtered_data['Média Móvel Intervalo'] = filtered_data['Preço'].rolling(window=3, min_periods=1).mean()  # Média móvel do intervalo

# Criar scatterplot com médias móveis
fig = go.Figure()

# Scatterplot para o intervalo selecionado
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Preço'], mode='markers', name='Preço (intervalo)'))
# Média móvel do intervalo
fig.add_trace(go.Scatter(x=filtered_data.index, y=filtered_data['Média Móvel Intervalo'], mode='lines', name='Média Móvel (Intervalo)'))
# Média móvel total

fig.update_layout(title='Preço e Média Móvel', xaxis_title='Data', yaxis_title='Preço em dólar', title_x=0.5)

# Formatar o índice para exibir apenas a data no formato dia, mês e ano
filtered_data.index = filtered_data.index.strftime('%Y-%m-%d')

# Exibir tabela dos dados filtrados
st.write(f"Exibindo dados para o intervalo de anos: {year_range[0]} - {year_range[1]}")
st.write(filtered_data)

# Exibir gráfico de scatterplot com médias móveis
st.plotly_chart(fig)

# Criar boxplot do intervalo selecionado
fig_box = px.box(filtered_data, y='Preço', title="Boxplot do Preço (Intervalo Selecionado)")
st.plotly_chart(fig_box)

# Garantir que o índice seja do tipo datetime
filtered_data.index = pd.to_datetime(filtered_data.index)

# Botão de seleção de frequência de agregação para o gráfico de candle
frequency = st.selectbox("Escolha a frequência para o gráfico de candle:", options=["Semanal", "Mensal", "Anual"], index=0)

# Ajustar a frequência com base na escolha do botão
if frequency == "Anual":
    ohlc_data = filtered_data.resample('Y').agg({
        'Preço': ['first', 'max', 'min', 'last']
    })
elif frequency == "Mensal":
    ohlc_data = filtered_data.resample('M').agg({
        'Preço': ['first', 'max', 'min', 'last']
    })
else:  # Semanal
    ohlc_data = filtered_data.resample('W').agg({
        'Preço': ['first', 'max', 'min', 'last']
    })

# Renomear colunas para o formato esperado no gráfico de candle
ohlc_data.columns = ['Open', 'High', 'Low', 'Close']

# Criar o gráfico de candle
fig_candle = go.Figure(data=[go.Candlestick(
    x=ohlc_data.index,
    open=ohlc_data['Open'],
    high=ohlc_data['High'],
    low=ohlc_data['Low'],
    close=ohlc_data['Close']
)])

# Configurações de layout do gráfico de candle
fig_candle.update_layout(
    title=f'Gráfico ({frequency}) para o Intervalo Selecionado',
    xaxis_title='Data',
    yaxis_title='Preço em dólar',
    title_x=0.5
)

# Exibir o gráfico de candle
st.plotly_chart(fig_candle)


st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('## Modelo', unsafe_allow_html=True)


def wmape(y_true, y_pred):
    
    # Converter para arrays do NumPy para facilitar o cálculo
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Soma dos erros absolutos
    numerador = np.sum(np.abs(y_true - y_pred))

    # soma dos valores reais
    denominador = np.sum(np.abs(y_true))

    #Calculo WMAPE
    valor_wmape = numerador/denominador

    return valor_wmape

st.markdown(
    """
    <style>
        .justified-text {
            text-align: justify;
        }
    </style>
    <div class="justified-text">
        Para os cálculos de previsão, foi utilizado o modelo AutoARIMA da biblioteca StatsForecast.
        Este modelo foi selecionado pela capacidade de ajuste automático dos parâmetros e pelo seu desempenho 
        superior em comparação com outros testados. No entanto, vale destacar que, 
        como o conjunto de dados é atualizado periodicamente a partir do site do IPEA, o modelo pode apresentar 
        variações em seu desempenho conforme mudanças sazonais nos dados. 
        Essas mudanças podem influenciar a capacidade do modelo de capturar padrões recorrentes, 
        impactando a precisão das previsões.<br><br>
        Para acompanhar a performance do modelo duas métricas serão utilizadas:<br><br>
        <strong>WMAPE (Weighted Mean Absolute Percentage Error)</strong><br>
        O WMAPE mede a precisão de previsões com base em um valor médio ponderado do erro percentual absoluto, 
        levando em conta o valor real como peso. O WMAPE ajudará a quantificar o erro em termos percentuais, 
        facilitando a interpretação sobre a precisão da previsão: valores mais baixos indicam maior precisão.<br><br>
        <strong>R² (Coeficiente de Determinação)</strong><br>
        Mostra a proporção da variação dos dados que é explicada pelo modelo de previsão.
        Quanto mais próximo de 1, melhor o modelo consegue se adaptar às variações dos dados observados, 
        o que indica um bom ajuste ao padrão dos dados históricos. Essa métrica será útil para ver o quanto o modelo 
        captura das variações reais nos preços do petróleo, conforme a interpretação do modelo.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<br><br>', unsafe_allow_html=True)

st.markdown('### Perfomance', unsafe_allow_html=True)
# Define a frequência do DataFrame para diário e preenche valores NaN com a média
df1 = df.asfreq('D')
media = df1.mean()
df1 = df1.fillna(media)

# Mantém apenas os últimos 6 meses de dados
data_mais_recente = df1.index.max()
tempo = data_mais_recente - pd.DateOffset(months=6)
df1 = df1.loc[tempo:data_mais_recente]

# Aplica logaritmo aos dados e, em seguida, calcula a diferença
df1 = np.log(df1)
df1 = df1.diff(1).dropna()  # Remove NaN resultantes da diferenciação

# Prepara o DataFrame para uso no modelo de previsão
df1 = df1.reset_index()
df1.rename(columns={'Data': 'ds', 'Preço': 'y'}, inplace=True)
df1['unique_id'] = 'Brent'
df1 = df1[['unique_id', 'ds', 'y']]

# Separação entre Dataset de treino e Dataset de teste
treino = df1.head(len(df1)-14)
teste = df1.tail(14)
h = teste['ds'].nunique()

sf_AA = StatsForecast(models=[AutoARIMA(season_length=7, stationary=True,seasonal=True)], freq='D', n_jobs=-1)
sf_AA.fit(treino)

previsao_AA = sf_AA.predict(h=h, level=[90])
previsao_AA = previsao_AA.reset_index().merge(teste, on=['ds', 'unique_id'], how='left')

# Filtrar para incluir apenas o último mês do treino
ultimo_mes = treino['ds'].max() - pd.DateOffset(months=1)
treino_filtrado = treino[treino['ds'] >= ultimo_mes]

# Criar o gráfico
plt.figure(figsize=(14, 7))

# Plotar o treino filtrado
plt.plot(treino_filtrado['ds'], treino_filtrado['y'], label="Treino (Último Mês)", color='blue', linestyle='--')

# Plotar os valores reais (teste)
plt.plot(previsao_AA['ds'], previsao_AA['y'], label="Teste", color='black', alpha=0.7)

# Plotar as previsões
plt.plot(previsao_AA['ds'], previsao_AA['AutoARIMA'], label="Previsão AutoARIMA", color='orange')

# Preencher o intervalo de confiança
plt.fill_between(
    previsao_AA['ds'],
    previsao_AA['AutoARIMA-lo-90'],
    previsao_AA['AutoARIMA-hi-90'],
    color='orange',
    alpha=0.2,
    label="Intervalo de Confiança 90%"
)

# Título e rótulos dos eixos
plt.title("Previsão do modelo com Intervalo de Confiança de 90%")
plt.xlabel("Data")
plt.ylabel(" ")

# Legenda e grade
plt.legend()
plt.grid(True)

# Exibir o gráfico no Streamlit
st.pyplot(plt)

# Apresentação das métricas de desempenho
precisao4_AA = wmape(previsao_AA['y'].values, previsao_AA['AutoARIMA'].values)
r24_AA = r2_score(previsao_AA['y'], previsao_AA['AutoARIMA'])

# HTML e CSS para estilizar as métricas
st.markdown(
    f"""
    <style>
        .metric-container {{
            display: flex;
            justify-content: space-around;
            margin-top: 20px;
        }}
        .metric-box {{
            background-color: #f0f2f6;
            padding: 20px;
            width: 45%;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            color: #333333;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #007BFF;
            margin-top: 8px;
        }}
    </style>

    <div class="metric-container">
        <div class="metric-box">
            <div class="metric-title">WMAPE</div>
            <div class="metric-value">{precisao4_AA:.2%}</div>
        </div>
        <div class="metric-box">
            <div class="metric-title">R²</div>
            <div class="metric-value">{r24_AA:.2f}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<br><br><br>', unsafe_allow_html=True)

st.markdown('### Previsão', unsafe_allow_html=True)

df_valores = df1[:-14]
df_previstos = df1[-14:]

modelo_AA = StatsForecast(models=[AutoARIMA(season_length=7, stationary=True,seasonal=True)], freq='D', n_jobs=-1)
modelo_AA.fit(df_valores)

previsao_AA = modelo_AA.predict(h=19, level=[90])
previsao_plot = previsao_AA[-6:]

# Filtrar os últimos 14 registros de df1
historico_14 = df1[-14:]

# Criar o gráfico
plt.figure(figsize=(14, 7))

# Plotar os dados históricos (últimos 14 valores)
plt.plot(historico_14['ds'], historico_14['y'], label="Histórico (Últimos 14 Valores)", color='blue', linestyle='--')

# Plotar as previsões
plt.plot(previsao_plot['ds'], previsao_plot['AutoARIMA'], label="Previsão AutoARIMA", color='orange', marker='o')

# Preencher o intervalo de confiança
plt.fill_between(
    previsao_plot['ds'],
    previsao_plot['AutoARIMA-lo-90'],
    previsao_plot['AutoARIMA-hi-90'],
    color='orange',
    alpha=0.2,
    label="Intervalo de Confiança 90%"
)

# Título e rótulos dos eixos
plt.title("Previsão AutoARIMA com Intervalo de Confiança de 90%")
plt.xlabel("Data")
plt.ylabel("Preço")

# Legenda e grade
plt.legend()
plt.grid(True)

# Exibir o gráfico no Streamlit
st.pyplot(plt)

def pipe(df):
    """Cria um pipeline e aplica transformações ao DataFrame fornecido."""
    
    # Criar o pipeline
    pipeline = Pipeline(steps=[
        ('dataframe_transformer', DataFrameTransformer()),  # Primeira transformação
        ('time_series_transformer', TimeSeriesTransformer())  # Segunda transformação
    ])
    
    # Aplicar o pipeline ao DataFrame
    series = pipeline.fit_transform(df)
    
    return series

previsao_futura = pipe(dff)

previsao_futura.index = previsao_futura.index.strftime('%Y-%m-%d')
st.write(previsao_futura)