import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Проверка наличия тяжелых библиотек
try:
    import seaborn as sns
    import plotly.graph_objects as go
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

# --- ВЫЧИСЛИТЕЛЬНОЕ ЯДРО ---
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
        w = np.random.uniform(-1, 1, (input_size, layer_size))
        if self.balanced and layer_size % 2 == 0:
            half = layer_size // 2
            w[:, half:] = -w[:, :half]
        self.weights = w * base_weight
    def forward(self, inputs):
        outputs = []
        for i, n in enumerate(self.neurons):
            off = np.pi if (self.balanced and i >= self.layer_size // 2) else 0
            net_in = np.dot(inputs, self.weights[:, i])
            outputs.append(n.s_transition(net_in + off))
        return np.array(outputs)

# --- ГЕОМЕТРИЯ ---
def get_coords(r, hc, hs, res=800):
    r_a = r / 2.0
    pts = []
    for i in range(int(res*0.3)+1):
        t = i / int(res*0.3)
        phi = np.pi * (1 - t)
        pts.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    for i in range(1, int(res*0.7)+1):
        t = i / int(res*0.7)
        pts.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), (hs/2) + hc*t])
    pts = np.array(pts)
    return np.vstack([-pts[::-1], pts])

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="FSIN Sync Fix", layout="wide")

if not HAS_LIBS:
    st.error("Ошибка: В requirements.txt не добавлены seaborn или plotly!")
    st.stop()

st.title("Синхронизированный комплекс ФСИН")

with st.sidebar:
    st.header("Настройки")
    mode = st.selectbox("Режим", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Нейроны", 2, 40, 10, step=2)
    wurf = st.number_input("Вурф", value=1.309)
    st.divider()
    r_c = st.slider("Радиус", 10, 100, 50)
    h_c = st.slider("Высота", 10, 100, 30)
    h_s = st.slider("S-переход", 5, 50, 10)

# Расчет
net = SfiralNetwork(2, nodes, wurf, balanced=(mode=="Антисимметричный (Резонанс)"))
t_ax = np.linspace(-10, 10, 400)
inp = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
res_states = np.array([net.forward(i) for i in inp])
bal = np.sum(res_states, axis=1)
s_idx = np.mean(np.abs(bal))

# Вывод
c1, c2 = st.columns([2, 1])
with c1:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_ax, res_states, alpha=0.2)
    ax.plot(t_ax, bal, color='black', lw=2)
    ax.axhline(0, color='red', ls='--')
    st.pyplot(fig)
with c2:
    st.metric("Stability Index", f"{s_idx:.18f}")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sns.heatmap(pd.DataFrame(res_states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax2)
    st.pyplot(fig2)

st.subheader("3D Сфираль")
c = get_coords(r_c, h_c, h_s)
f3d = go.Figure(data=[go.Scatter3d(x=c[:,0], y=c[:,1], z=c[:,2], mode='lines', line=dict(color='red', width=5))])
f3d.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(f3d, use_container_width=True)

st.download_button("Скачать CSV", pd.DataFrame(res_states, columns=[f"Node_{i+1}" for i in range(nodes)]).to_csv(index=False).encode('utf-8'), "fsin_sphiral.csv")
