import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- ЯДРО ФСИН ---
class SfiralNeuron:
    def __init__(self, frequency=1.236, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        f_loop = np.sin(self.frequency * x + self.phase_shift)
        b_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(f_loop + b_loop)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        weights = np.random.uniform(-1, 1, (input_size, layer_size))
        if self.balanced and layer_size % 2 == 0:
            half = layer_size // 2
            weights[:, half:] = -weights[:, :half]
        self.weights = weights * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            p_offset = np.pi if (self.balanced and i >= self.layer_size // 2) else 0
            current_phase = neuron.phase_shift + p_offset
            net_input = np.dot(inputs, self.weights[:, i])
            f_loop = np.sin(neuron.frequency * net_input + current_phase)
            b_loop = -np.sin(neuron.frequency * net_input - current_phase)
            outputs.append(np.tanh(f_loop + b_loop))
        return np.array(outputs)

# --- ГЕОМЕТРИЯ СФИРАЛИ ---
def generate_sfiral_coords(r_coil, h_coil, h_s, turns=1.0, res=1000):
    r_arc = r_coil / 2.0
    right_p = []
    res_arc = int(res * 0.3)
    for i in range(res_arc + 1):
        t = i / res_arc
        phi = np.pi * (1 - t)
        x = r_arc + r_arc * np.cos(phi)
        y = -r_arc * np.sin(phi)
        z = (h_s / 2) * t
        right_p.append([x, y, z])
    res_coil = int(res * 0.7)
    z_start = h_s / 2
    for i in range(1, res_coil + 1):
        t = i / res_coil
        theta = turns * 2 * np.pi * t
        x = r_coil * np.cos(theta)
        y = r_coil * np.sin(theta)
        z = z_start + (h_coil * t)
        right_p.append([x, y, z])
    right_p = np.array(right_p)
    left_p = -right_p[::-1]
    return np.vstack([left_p, right_p])

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="FSIN Analyzer Pro", layout="wide")
st.title("Комплекс ФСИН: Моделирование и Анализ Антисимметрии")

st.sidebar.header("Геометрия и Веса")
r_val = st.sidebar.slider("Радиус (R_coil)", 10.0, 100.0, 50.0)
h_coil = st.sidebar.slider("Высота витков", 10.0, 100.0, 30.0)
h_s = st.sidebar.slider("Высота S-узла", 5.0, 50.0, 10.0)
wurf = st.sidebar.number_input("Золотой вурф", value=1.309, format="%.3f")

st.sidebar.header("Параметры Сети")
mode = st.sidebar.selectbox("Режим", ["Антисимметричный (Резонанс)", "Случайный"])
nodes = st.sidebar.number_input("Нейроны", 2, 40, 10, step=2)

# Вычисления
net = SfiralNetwork(2, nodes, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
x_ax = np.linspace(-10, 10, 400)
inputs = np.column_stack([np.sin(x_ax), np.cos(x_ax)])
states = np.array([net.forward(inp) for inp in inputs])
total_bal = np.sum(states, axis=1)
stab_idx = np.mean(np.abs(total_bal))

# --- ГРАФИКИ И ТЕРМОКАРТА ---
col_main, col_stats = st.columns([2, 1])

with col_main:
    st.subheader("Динамика Баланса")
    fig_dyn, ax_dyn = plt.subplots(figsize=(10, 4))
    ax_dyn.plot(x_ax, states, alpha=0.2)
    ax_dyn.plot(x_ax, total_bal, color='black', linewidth=2, label='Total Balance')
    ax_dyn.axhline(0, color='red', linestyle='--')
    st.pyplot(fig_dyn)

with col_stats:
    st.metric("Stability Index", f"{stab_idx:.18f}")
    st.subheader("Матрица Корреляции (Термокарта)")
    fig_corr, ax_corr = plt.subplots(figsize=(5, 5))
    df_corr = pd.DataFrame(states, columns=[f"N{i+1}" for i in range(nodes)])
    sns.heatmap(df_corr.corr(), cmap='RdBu_r', center=0, ax=ax_corr, cbar=False)
    st.pyplot(fig_corr)

# 3D ПРОЕКЦИЯ
st.subheader("Геометрия Сфирали")
coords = generate_sfiral_coords(r_val, h_coil, h_s)
fig3d = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='lines', line=dict(color='red', width=6))])
fig3d.add_trace(go.Scatter3d(x=[-r_val*1.5, r_val*1.5], y=[0,0], z=[0,0], mode='lines', line=dict(color='white', width=2, dash='dash')))
fig3d.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig3d, use_container_width=True)

# ЭКСПОРТ
st.download_button("Скачать fsin_sphiral.csv", df_corr.to_csv(index=False).encode('utf-8'), "fsin_sphiral.csv")
