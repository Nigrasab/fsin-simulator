import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- ЯДРО ФСИН (РЕЗОНАНС) ---
class SfiralNeuron:
    def __init__(self, frequency=1.236, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift
    def s_transition(self, x):
        f = np.sin(self.frequency * x + self.phase_shift)
        b = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(f + b)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        # Генерация базовых весов
        w = np.random.uniform(-1, 1, (input_size, layer_size))
        self.weights = w * base_weight
    def forward(self, inputs):
        outputs = []
        for i, n in enumerate(self.neurons):
            net_in = np.dot(inputs, self.weights[:, i])
            val = n.s_transition(net_in)
            # Прямая антисимметрия: вторая половина нейронов инвертирует первую
            if self.balanced and i >= self.layer_size // 2:
                pair_idx = i - self.layer_size // 2
                # Берем значение зеркального нейрона и инвертируем его
                val = -outputs[pair_idx] 
            outputs.append(val)
        return np.array(outputs)

# --- ГЕОМЕТРИЯ (ИЗ ВАШЕГО СКРИПТА BLENDER) ---
def generate_sfiral(r, hc, hs, res=1000):
    r_a, p_r = r / 2.0, []
    # 1. S-Дуга
    for i in range(int(res*0.3)+1):
        t = i / int(res*0.3)
        phi = np.pi * (1 - t)
        p_r.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    # 2. Виток
    for i in range(1, int(res*0.7)+1):
        t = i / int(res*0.7)
        p_r.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), (hs/2) + hc*t])
    p_r = np.array(p_r)
    return np.vstack([-p_r[::-1], p_r]) # Зеркальное отражение

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="FSIN Resonator Pro", layout="wide")
st.title("Сфиральный Резонатор: Точка Гармонии")

with st.sidebar:
    st.header("Настройки")
    mode = st.selectbox("Режим", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Нейроны (четное)", 2, 40, 10, step=2)
    wurf = st.number_input("Вурф", value=1.309, format="%.3f")
    st.divider()
    r_c = st.slider("Радиус R", 10, 100, 50)
    h_c = st.slider("Высота витка", 10, 100, 30)
    h_s = st.slider("Высота S-узла", 5, 50, 10)

# Вычисления
is_res = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(2, nodes, wurf, balanced=is_res)
t_ax = np.linspace(-10, 10, 400)
inp = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
res_states = np.array([net.forward(i) for i in inp])
bal = np.sum(res_states, axis=1)
s_idx = np.mean(np.abs(bal))

# Графики
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Фазовый Баланс")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_ax, res_states, alpha=0.2, lw=1)
    ax.plot(t_ax, bal, color='black', lw=2.5, label='Суммарный вектор')
    ax.axhline(0, color='red', ls='--', alpha=0.5)
    st.pyplot(fig)
with c2:
    st.metric("Stability Index", f"{s_idx:.18f}")
    st.subheader("Термокарта")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sns.heatmap(pd.DataFrame(res_states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax2)
    st.pyplot(fig2)

st.subheader("3D Сфираль (S-Геометрия)")
coords = generate_sfiral(r_c, h_c, h_s)
f3d = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='lines', line=dict(color='red', width=6))])
f3d.add_trace(go.Scatter3d(x=[-r_c*1.5, r_c*1.5], y=[0,0], z=[0,0], mode='lines', line=dict(color='white', width=2, dash='dash')))
f3d.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(f3d, use_container_width=True)

st.download_button("Скачать fsin_sphiral.csv", pd.DataFrame(res_states, columns=[f"Node_{i+1}" for i in range(nodes)]).to_csv(index=False).encode('utf-8'), "fsin_sphiral.csv")
