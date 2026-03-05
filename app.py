import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- 1. ЯДРО ФСИН (МАТЕМАТИКА РЕЗОНАНСА) ---
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
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size)) * base_weight
    def forward(self, inputs):
        outputs = []
        for i, n in enumerate(self.neurons):
            net_in = np.dot(inputs, self.weights[:, i])
            val = n.s_transition(net_in)
            if self.balanced and i >= self.layer_size // 2:
                val = -outputs[i - self.layer_size // 2] 
            outputs.append(val)
        return np.array(outputs)

# --- 2. ГЕОМЕТРИЯ СФИРАЛИ (BLENDER S-УЗЕЛ) ---
def generate_sfiral(r, hc, hs, res=600):
    r_a, pts = r / 2.0, []
    for i in range(int(res*0.3)+1):
        t = i / int(res*0.3)
        phi = np.pi * (1 - t)
        pts.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    for i in range(1, int(res*0.7)+1):
        t = i / int(res*0.7)
        pts.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), (hs/2) + hc*t])
    pts = np.array(pts)
    return np.vstack([-pts[::-1], pts])

# --- 3. МОДЕЛИРОВАНИЕ РОБОТА-НАСЕКОМОГО ---
def get_robot_insect_data(wurf, body_l=217.6, body_w=166.2):
    # Размеры сегментов ноги по Вурфу
    l1 = 88.0  # Femur
    l2 = l1 / wurf  # Tibia
    l3 = l2 / wurf  # Tarsus
    
    traces = []
    # Корпус (прямоугольник)
    bx = [body_l/2, body_l/2, -body_l/2, -body_l/2, body_l/2]
    by = [body_w/2, -body_w/2, -body_w/2, body_w/2, body_w/2]
    bz = [0, 0, 0, 0, 0]
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='silver', width=8), name="Корпус"))

    # Ноги (6 штук)
    side = [1, 1, 1, -1, -1, -1]
    angles = [np.pi/4, 0, -np.pi/4, np.pi/4, 0, -np.pi/4]
    
    for i in range(6):
        start_x = (body_l/2.5) if i in [0, 3] else (0 if i in [1, 4] else -body_l/2.5)
        start_y = (body_w/2) * side[i]
        
        # Точки сегментов
        ang = angles[i] + (0 if side[i]>0 else np.pi)
        p1 = [start_x + l1*np.cos(ang), start_y + l1*np.sin(ang), -20]
        p2 = [p1[0] + l2*np.cos(ang), p1[1] + l2*np.sin(ang), -60]
        p3 = [p2[0] + l3*np.cos(ang), p2[1] + l3*np.sin(ang), -100]
        
        lx = [start_x, p1[0], p2[0], p3[0]]
        ly = [start_y, p1[1], p2[1], p3[1]]
        lz = [0, p1[2], p2[2], p3[2]]
        
        traces.append(go.Scatter3d(x=lx, y=ly, z=lz, mode='lines+markers', 
                                   line=dict(color='gold' if side[i]>0 else 'orange', width=6),
                                   marker=dict(size=4), name=f"Нога {i+1}"))
    return traces

# --- 4. ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="FSIN Biomorphic Robot", layout="wide")
st.title("ФСИН: Интеграция Резонанса в Робототехнику")

tab1, tab2 = st.tabs(["📊 Анализ Сигналов", "🐜 3D Модель Робота"])

with st.sidebar:
    st.header("Константы")
    mode = st.selectbox("Режим", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Нейроны", 2, 40, 10, step=2)
    wurf = st.number_input("Золотой вурф", value=1.309, format="%.3f")
    st.divider()
    st.header("Геометрия (мм)")
    r_c = st.slider("Радиус R", 10, 100, 50)
    h_c = st.slider("Высота", 10, 100, 30)
    h_s = st.slider("S-переход", 5, 50, 10)

# Расчеты сети
net = SfiralNetwork(2, nodes, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
t_ax = np.linspace(-10, 10, 400)
inp = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
states = np.array([net.forward(i) for i in inp])
total_bal = np.sum(states, axis=1)
stab_idx = np.mean(np.abs(total_bal))

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Фазовая устойчивость")
        fig2d, ax2d = plt.subplots(figsize=(10, 4))
        ax2d.plot(t_ax, states, alpha=0.2); ax2d.plot(t_ax, total_bal, color='black', lw=2)
        ax2d.axhline(0, color='red', ls='--'); st.pyplot(fig2d)
    with c2:
        st.metric("Stability Index", f"{stab_idx:.18f}")
        fig_h, ax_h = plt.subplots(figsize=(5, 5))
        sns.heatmap(pd.DataFrame(states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)
    
    st.subheader("Геометрия S-узла (Blender Script)")
    coords = generate_sfiral(r_c, h_c, h_s)
    fig3s = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='lines', line=dict(color='red', width=6))])
    fig3s.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
    st.plotly_chart(fig3s, use_container_width=True)

with tab2:
    st.subheader("Цифровой двойник: Сфиральный Гексапод")
    st.markdown(f"Пропорции сегментов ног рассчитаны по Вурфу ({wurf}). Габариты корпуса: 217.6 x 166.2 мм.")
    robot_traces = get_robot_insect_data(wurf)
    fig_robot = go.Figure(data=robot_traces)
    fig_robot.update_layout(scene=dict(aspectmode='data', 
                                       xaxis=dict(range=[-300, 300]), 
                                       yaxis=dict(range=[-300, 300]), 
                                       zaxis=dict(range=[-150, 150])),
                            margin=dict(l=0, r=0, b=0, t=0), height=700)
    st.plotly_chart(fig_robot, use_container_width=True)

st.download_button("Скачать данные", pd.DataFrame(states).to_csv(index=False).encode('utf-8'), "fsin_sphiral.csv")
