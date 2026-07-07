import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(page_title="Resultados do Filtro Prensa", layout="wide")

# Título Principal
st.title("📊 Resultados e Simulação - Filtro Prensa")
st.markdown("---")

# Criação das Abas principais
aba1, aba2 = st.tabs(["📊 Dados Experimentais", "🔬 Análise Teórica (Manual)"])

# Constantes fixas da bancada XP1613.2 de acordo com o manual
area_filtracao = 0.3128  
concentracao = 60.0

# ==========================================
# ABA 1: DADOS EXPERIMENTAIS (Com Linha de Tendência)
# ==========================================
with aba1:
    st.sidebar.header("⚙️ Seleção do Ensaio Experimental")

    # Escolha da pressão do experimento real
    pressao_nominal = st.sidebar.radio(
        "Escolha a pressão do ensaio:",
        [2, 3],
        format_func=lambda x: f"{x} mca"
    )

    if pressao_nominal == 2:
        dados = {
            't (s)': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            'V (mm)': [40, 70, 85, 95, 103, 115, 124, 132, 139, 148, 155, 163, 169],
            'Q (m³/h)': [0.4, 0.33, 0.42, 0.43, 0.43, 0.43, 0.41, 0.43, 0.43, 0.44, 0.45, 0.47, 0.49],
            'V (m³)': [0.00192, 0.00336, 0.00408, 0.00456, 0.00494, 0.00552, 0.00595, 0.00633, 0.00667, 0.00710, 0.00744, 0.00782, 0.00811]
        }
    else:
        dados = {
            't (s)': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
            'V (mm)': [30, 45, 54, 60, 67, 75, 83, 89, 95, 103, 108, 113, 119],
            'Q (m³/h)': [0.21, 0.21, 0.2, 0.19, 0.18, 0.18, 0.17, 0.21, 0.27, 0.27, 0.28, 0.35, 0.34],
            'V (m³)': [0.00144, 0.00216, 0.00259, 0.00288, 0.00321, 0.00360, 0.00398, 0.00427, 0.00456, 0.00494, 0.00518, 0.00542, 0.00571]
        }

    df = pd.DataFrame(dados)

    t_s = df['t (s)'].iloc[0]
    V_s = df['V (m³)'].iloc[0]

    df['Delta t'] = df['t (s)'] - t_s
    df['Delta V'] = df['V (m³)'] - V_s

    df['tempo por unidade de volume'] = np.where(df['Delta V'] != 0, df['Delta t'] / df['Delta V'], np.nan)
    df_valido = df.dropna().copy()

    coefs = np.polyfit(df_valido['V (m³)'], df_valido['tempo por unidade de volume'], 1)
    angular_a = coefs[0]
    linear_b = coefs[1]

    p = np.poly1d(coefs)
    y_pred = p(df_valido['V (m³)'])
    y_actual = df_valido['tempo por unidade de volume']
    r_quadrado = 1 - (np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2))

    st.info(f"Exibindo análises calculadas para o regime de **Pressão Constante** a **{pressao_nominal} mca**.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Coeficiente Angular (a)", f"{angular_a:.2e} s/m⁶")
    col2.metric("Coeficiente Linear (b)", f"{linear_b:.2e} s/m³")
    col3.metric("Coeficiente de Ajuste (R²)", f"{r_quadrado:.4f}")

    st.markdown(f"### Perfil de Filtração - Linha de Tendência Ajustada para {pressao_nominal} mca")
    
    # --- CONSTRUÇÃO DO GRÁFICO COM LINHA DE TENDÊNCIA (Matplotlib) ---
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Pontos Experimentais (Azul)
    ax.scatter(df_valido['V (m³)'], df_valido['tempo por unidade de volume'], color='#1f77b4', s=50, label='Dados Experimentais')
    
    # Linha de Tendência (Vermelha Tracejada)
    x_linha = np.linspace(df_valido['V (m³)'].min(), df_valido['V (m³)'].max(), 100)
    y_linha = angular_a * x_linha + linear_b
    ax.plot(x_linha, y_linha, color='red', linestyle='--', linewidth=2, label=f'Ajuste Linear (R² = {r_quadrado:.4f})')
    
    # Customização de eixos e legenda
    ax.set_xlabel("Volume Coletado V (m³)", fontsize=10)
    ax.set_ylabel("(t - ts) / (V - Vs)  (s/m³)", fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=10)
    
    # Renderiza o gráfico no Streamlit
    st.pyplot(fig)

    st.markdown("### 📋 Dados Experimentais Tratados")
    st.dataframe(df[['t (s)', 'V (mm)', 'Q (m³/h)', 'V (m³)', 'tempo por unidade de volume']], use_container_width=True)


# ==========================================
# ABA 2: ANÁLISE TEÓRICA
# ==========================================
with aba2:
    st.subheader("🔬 Simulador Baseado nas Equações do Manual")
    st.markdown("Ajuste os parâmetros operacionais abaixo para gerar a curva e os volumes teóricos de filtração para o intervalo de **1 minuto**:")

    c1, c2, c3 = st.columns(3)
    p_teorica_mca = c1.number_input("Pressão de Operação desejada (mca):", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
    vazao_inicial_teorica = c2.number_input("Vazão de Projeto Inicial (m³/h):", min_value=0.05, max_value=2.0, value=0.4, step=0.05)
    viscosidade = c3.number_input("Viscosidade do Fluido (Pa·s):", value=1.0e-3, format="%.2e")

    c4, c5 = st.columns(2)
    alpha_teorico = c4.number_input("Resistência Específica da Torta (α) [m/kg]:", value=2.0e11, format="%.2e")
    R_teorico = c5.number_input("Resistência do Meio Filtrante (R) [m⁻¹]:", value=5.0e10, format="%.2e")

    delta_p_pa = p_teorica_mca * 9806.65

    a_teorico = (alpha_teorico * viscosidade * concentracao) / (2 * (area_filtracao ** 2) * delta_p_pa)
    b_teorico = (viscosidade * R_teorico) / (area_filtracao * delta_p_pa)

    tempos_teoricos = np.arange(0, 65, 5)
    volumes_teoricos = []

    for t in tempos_teoricos:
        if t == 0:
            volumes_teoricos.append(0.0)
        else:
            v_calc = (-b_teorico + np.sqrt((b_teorico ** 2) - 4 * a_teorico * (-t))) / (2 * a_teorico)
            volumes_teoricos.append(v_calc)

    df_teorico = pd.DataFrame({
        'Tempo t (s)': tempos_teoricos,
        'Volume Teórico V (m³)': volumes_teoricos
    })

    df_teorico['t/V Teórico (s/m³)'] = np.where(df_teorico['Volume Teórico V (m³)'] != 0, 
                                                df_teorico['Tempo t (s)'] / df_teorico['Volume Teórico V (m³)'], 
                                                np.nan)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown("**Curva Teórica de Volume Acumulado**")
        st.line_chart(data=df_teorico, x='Tempo t (s)', y='Volume Teórico V (m³)', use_container_width=True)
        
    with col_g2:
        st.markdown("**Gráfico de Linearização Teórica (Svarovsky)**")
        st.line_chart(data=df_teorico.dropna(), x='Volume Teórico V (m³)', y='t/V Teórico (s/m³)', use_container_width=True)

    st.markdown("### 📋 Tabela de Valores Calculados Teoricamente")
    st.dataframe(df_teorico, use_container_width=True)
