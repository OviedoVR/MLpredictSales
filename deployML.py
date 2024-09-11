import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
import pickle

st.header('Estimando vendas futuras com Machine Learning')
st.image('assets/vendas.png')

st.sidebar.markdown('''
Uma empresa de marketing digital deseja otimizar seus investimentos em publicidade em diferentes canais (TV, rádio e jornal) para maximizar as vendas de seus produtos. A empresa possui dados históricos sobre o investimento em cada canal e as vendas correspondentes.

**Objetivo:** utilizar técnicas de Machine Learning para construir um modelo preditivo que estime as vendas futuras com base nos investimentos em publicidade.

''')

st.subheader('Dados')

st.markdown('''
Metadados do dataset a ser utilizado:

- `TV`: investimento em publicidade em televisão para um determinado produto ou campanha (USD)
- `Radio`:  investimento em publicidade em rádio (USD).
- `Newspaper`: investimento em publicidade em jornais (USD).
- `Sales`:  resultado final da campanha publicitária, ou seja, as vendas geradas.
''')

# Carregar dados e modelo
dados = pd.read_csv('assets/campanhas_publicidade.csv')
with open('assets/modelo_regressao_00.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Criar o scaler com base nos dados
scaler = StandardScaler()
scaler.fit(dados[['TV', 'Radio', 'Newspaper']])

st.markdown('---')
st.subheader('Predição via Regressão Linear Múltipla')

# Entrada do usuário
TV = st.slider('Valor gasto em TV (USD)', 0, 120, step=5)
Radio = st.slider('Valor gasto em Radio (USD)', 0, 100, step=5)
Newspaper = st.slider('Valor gasto em jornal (USD)', 0, 100, step=5)
label = st.selectbox('Objetivo da campanha', dados['Label'].sort_values().unique())

# Criar DataFrame para predição
input_data = pd.DataFrame({
    'TV': [TV],
    'Radio': [Radio],
    'Newspaper': [Newspaper],
    'Label_Branding': [1 if label == "Branding" else 0],
    'Label_Sales': [1 if label == "Sales" else 0]
})

# Aplicar o escalonamento aos dados de entrada
input_data[['TV', 'Radio', 'Newspaper']] = scaler.transform(input_data[['TV', 'Radio', 'Newspaper']])

# Botão para realizar a previsão
if st.button('Predizer'):
    resultado = modelo.predict(input_data)
    resultado = (resultado[0].item())
    st.markdown(f"### Vendas: {int(resultado)}")


st.markdown('---')
st.subheader('Métricas de desempenho')
st.markdown('''
 * **R²:** 91,4% 
 * **RMSE:** 1,50
''')

st.image('assets/performance.png')
st.markdown('> **Versão 1** -- modelo baseline - Set. 2024')
