import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- 1. МАТЕМАТИЧЕСКОЕ ЯДРО (ФСИН) ---
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
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, layer_size)) * base_weight
    def forward(self, inputs):
        outputs = []
        for i, n in enumerate(self.neurons):
            net_in = np.dot(inputs, self.weights[:, i])
            val = n.s_transition(net_in)
            if self.balanced and i >= self.layer_size // 2:
                val = -outputs[i - self.layer_size // 2] 
            outputs.append(val)
        return np.array(outputs)

# --- 2. ГЕОМЕТРИЯ СФИРАЛИ (S-УЗЕЛ) ---
def generate_sfiral_coords(r, hc, hs, res=800):
    r_a, pts = r / 2.0, []
    res_arc = int(res * 0.3)
    for i in range(res_arc + 1):
        t = i / res_arc
        phi = np.pi * (1 - t)
        pts.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    res_coil = int(res * 0.7)
    z_start = hs / 2
    for i in range(1, res_coil + 1):
        t = i / res_coil
        pts.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), z_start + hc*t])
    pts = np.array(pts)
    left_p = -pts[::-1]
    return np.vstack([left_p, pts])

# --- 3. КИНЕМАТИКА РОБОТА ---
def calculate_leg_points(start_xyz, base_angle, lift_val, stretch_val, wurf, r_base):
    l_coxa = r_base * 0.4
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    gamma = base_angle + (stretch_val * 0.2)
    beta = 0.5 + (lift_val * 0.3)
    alpha = 1.2 + (lift_val * 0.2)
    p1 = [
        start_xyz[0] + l_femur * np.cos(gamma) * np.cos(beta),
        start_xyz[1] + l_femur * np.sin(gamma) * np.cos(beta),
        start_xyz[2] + l_femur * np.sin(beta)
    ]
    p2 = [
        p1[0] + l_tibia * np.cos(gamma) * np.cos(beta - alpha),
        p1[1] + l_tibia * np.sin(gamma) * np.cos(beta - alpha),
        p1[2] + l_tibia * np.sin(beta - alpha)
    ]
    return [start_xyz, p1, p2]

# --- 4. ИНТЕРФЕЙС ---
st.set_page_config(page_title="FSIN Integrated Lab", layout="wide")
st.title("Интегрированный комплекс ФСИН: Анализ и Робототехника")

with st.sidebar:
    st.header("1. Параметры Сети")
    mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Узлы (четное)", 6, 40, 10, step=2)
    wurf = st.number_input("Золотой вурф", 1.0, 2.0, 1.309, format="%.3f")
    
    st.divider()
    st.header("2. Геометрия Сфирали")
    r_c = st.slider("Радиус (R_coil)", 10, 100, 50)
    h_c = st.slider("Высота витка", 10, 100, 30)
    h_s = st.slider("Высота S-узла", 5, 50, 10)
    
    st.divider()
    st.header("3. Параметры Робота")
    body_l = st.slider("Длина корпуса", 100, 300, 217)
    body_w = st.slider("Ширина корпуса", 50, 250, 166)
    t_phase = st.slider("Фаза движения (t)", 0.0, 10.0, 0.0)

# Расчеты
net = SfiralNetwork(2, nodes, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
t_ax = np.linspace(-10, 10, 400)
inp_signals = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
states = np.array([net.forward(i) for i in inp_signals])
total_bal = np.sum(states, axis=1)
stab_idx = np.mean(np.abs(total_bal))

# ВКЛАДКИ
tab1, tab2, tab3 = st.tabs(["📊 Анализ Резонанса", "🌀 Геометрия Узла", "🕷️ Биоморфный Робот"])

with tab1:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Осциллограмма состояний")
        fig2d, ax2d = plt.subplots(figsize=(10, 4))
        ax2d.plot(t_ax, states, alpha=0.3)
        ax2d.plot(t_ax, total_bal, color='black', lw=2, label='Суммарный вектор')
        ax2d.axhline(0, color='red', ls='--')
        st.pyplot(fig2d)
    with col_b:
        st.metric("Stability Index", f"{stab_idx:.18f}")
        st.subheader("Термокарта")
        fig_h, ax_h = plt.subplots(figsize=(5, 5))
        sns.heatmap(pd.DataFrame(states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab2:
    st.subheader("3D Проекция S-узла (Blender Script)")
    coords = generate_sfiral_coords(r_c, h_c, h_s)
    fig3s = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], 
                                         mode='lines', line=dict(color='red', width=6))])
    fig3s.add_trace(go.Scatter3d(x=[-r_c*1.5, r_c*1.5], y=[0,0], z=[0,0], 
                                 mode='lines', line=dict(color='white', width=2, dash='dash')))
    fig3s.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig3s, use_container_width=True)

with tab3:
    st.subheader("Цифровой двойник Гексапода")
    curr_state = net.forward(t_phase)
    fig_r = go.Figure()
    # Корпус
    t_b = np.linspace(0, 2*np.pi, 50)
    fig_r.add_trace(go.Scatter3d(x=(body_l/2)*np.cos(t_b), y=(body_w/2)*np.sin(t_b), z=t_b*0, 
                                 mode='lines', line=dict(color='silver', width=10), name="Корпус"))
    # Ноги
    m_angles = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    for i in range(6):
        m_ang = m_angles[i]
        start = [(body_l/2)*np.cos(m_ang), (body_w/2)*np.sin(m_ang), 0]
        lift, stretch = curr_state[i % nodes], curr_state[(i+1) % nodes]
        pts = np.array(calculate_leg_points(start, m_ang, lift, stretch, wurf, r_c))
        fig_r.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines+markers',
                                     line=dict(color='gold' if i<3 else 'orange', width=8), name=f"Нога {i+1}"))
    fig_r.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-150, 150])), margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig_r, use_container_width=True)

st.download_button("Экспорт CSV", pd.DataFrame(states).to_csv(index=False).encode('utf-8'), "fsin_full_data.csv")
