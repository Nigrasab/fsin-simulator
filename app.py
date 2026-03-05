import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="FSIN: Сквозное моделирование", layout="wide")

# --- 1. МАТЕМАТИЧЕСКИЙ ГЕНОМ ---
class SfiralNeuron:
    def __init__(self, frequency=1.236, phase_shift=np.pi/2):
        self.freq = frequency
        self.phase = phase_shift
        
    def s_transition(self, x):
        f = np.sin(self.freq * x + self.phase)
        b = -np.sin(self.freq * x - self.phase)
        return np.tanh(f + b)

class SfiralNetwork:
    def __init__(self, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, (2, layer_size)) * base_weight
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]

    def get_states(self, t_val):
        x = np.sin(t_val)
        y = np.cos(t_val)
        net_in = np.dot(np.array([x, y]), self.weights)
        
        raw = np.array([self.neurons[i].s_transition(net_in[i]) for i in range(self.layer_size)])
        
        if self.balanced:
            half = self.layer_size // 2
            raw[half:] = -raw[:half]
        return raw

# --- 2. ГЕОМЕТРИЧЕСКИЙ ПРОЕКТОР ---
def build_sfiral(r, hc, hs):
    res = 600
    r_a = r / 2.0
    pts = []
    
    for i in range(int(res * 0.3) + 1):
        t = i / (res * 0.3)
        phi = np.pi * (1 - t)
        pts.append([r_a + r_a * np.cos(phi), -r_a * np.sin(phi), (hs / 2) * t])
        
    for i in range(1, int(res * 0.7) + 1):
        t = i / (res * 0.7)
        pts.append([r * np.cos(2 * np.pi * t), r * np.sin(2 * np.pi * t), (hs / 2) + hc * t])
        
    pts = np.array(pts)
    left = -np.copy(pts[::-1])
    return np.vstack([left, pts])

def build_insect_legs(wurf, states, body_l, body_w, r_base):
    l_coxa = r_base * 0.3
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    traces = []

    t_b = np.linspace(0, 2 * np.pi, 50)
    bx = (body_l / 2) * np.cos(t_b)
    by = (body_w / 2) * np.sin(t_b)
    bz = np.zeros_like(t_b)
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='silver', width=12), name="Корпус"))

    m_angs = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    
    for i in range(6):
        m_a = m_angs[i]
        sx = (body_l / 2) * np.cos(m_a)
        sy = (body_w / 2) * np.sin(m_a)
        
        s = states[i % len(states)]
        ang = m_a + (s * 0.2)
        
        p1x = sx + l_femur * np.cos(ang)
        p1y = sy + l_femur * np.sin(ang)
        p1z = 25 * s
        
        p2x = p1x + l_tibia * np.cos(ang)
        p2y = p1y + l_tibia * np.sin(ang)
        p2z = -60
        
        traces.append(go.Scatter3d(x=[sx, p1x, p2x], y=[sy, p1y, p2y], z=[0, p1z, p2z],
                                   mode='lines+markers', line=dict(color='#ffca28', width=8), name=f"Leg {i+1}"))
    return traces

# --- 3. ИНТЕРФЕЙС И ГЕНЕТИЧЕСКИЙ КОД ---
st.title("Комплекс ФСИН: Сквозное моделирование")

with st.sidebar:
    st.header("🧬 Генетический код")
    mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Узлы (четное)", 6, 40, 10, step=2)
    wurf = st.number_input("Золотой вурф", 1.0, 2.0, 1.309, format="%.3f")
    
    st.subheader("Форма (Сфираль)")
    r_c = st.slider("Радиус каркаса", 10, 100, 50)
    h_c = st.slider("Высота витков", 10, 100, 30)
    h_s = st.slider("Высота S-узла", 5, 50, 10)
    
    st.subheader("Жизнь (Робот)")
    b_l = st.slider("Длина тела", 100, 300, 217)
    b_w = st.slider("Ширина тела", 50, 250, 166)
    t_phase = st.slider("Фаза движения", 0.0, 10.0, 0.0)

# --- 4. РАСЧЕТЫ ---
is_bal = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(nodes, wurf, balanced=is_bal)

t_range = np.linspace(0, 10, 300)
all_states = np.array([net.get_states(t) for t in t_range])
current_states = net.get_states(t_phase)
stab_idx = np.mean(np.abs(np.sum(all_states, axis=1)))

# --- 5. ВИЗУАЛИЗАЦИЯ (ВКЛАДКИ) ---
tab1, tab2, tab3 = st.tabs(["📊 Уровень I: ЭНЕРГИЯ", "🌀 Уровень II: ФОРМА", "🕷️ Уровень III: ЖИЗНЬ"])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t_range, all_states, alpha=0.3)
        ax1.plot(t_range, np.sum(all_states, axis=1), color='red', lw=2)
        st.pyplot(fig1)
    with c2:
        st.metric("Индекс устойчивости", f"{stab_idx:.18f}")
        fig_h, ax_h = plt.subplots(figsize=(5, 5))
        sns.heatmap(pd.DataFrame(all_states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab2:
    coords = build_sfiral(r_c, h_c, h_s)
    fig2 = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], 
                                        mode='lines', line=dict(color='#ffca28', width=6))])
    fig2.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    traces = build_insect_legs(wurf, current_states, b_l, b_w, r_c)
    fig3 = go.Figure(data=traces)
    fig3.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-100, 100])), margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig3, use_container_width=True)

df_export = pd.DataFrame(all_states, columns=[f"Node_{i+1}" for i in range(nodes)])
st.download_button("Экспорт CSV", df_export.to_csv(index=False).encode('utf-8'), "fsin_genetic_matrix.csv")
