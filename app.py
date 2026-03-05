import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- 1. ВЫЧИСЛИТЕЛЬНОЕ ЯДРО (ФСИН) ---
class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        self.weights = np.random.uniform(-0.5, 0.5, (input_size, layer_size)) * base_weight
    def forward(self, t):
        # Генерируем волновой сигнал на базе S-перехода
        x = np.sin(1.236 * t)
        y = np.cos(1.236 * t)
        raw = np.tanh(np.dot([x, y], self.weights))
        if self.balanced:
            half = self.layer_size // 2
            raw[half:] = -raw[:half] # Идеальная антисимметрия
        return raw

# --- 2. КИНЕМАТИКА БИОМОРФНОЙ НОГИ ---
def calculate_leg_points(start_xyz, base_angle, lift_val, stretch_val, wurf, r_base):
    # Длины сегментов по Вурфу
    l_coxa = r_base * 0.4
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    
    # Углы суставов (биоморфные)
    gamma = base_angle + (stretch_val * 0.2) # Поворот (Z)
    beta = 0.5 + (lift_val * 0.3)            # Подъем бедра (XY)
    alpha = 1.2 + (lift_val * 0.2)           # Изгиб голени (вниз)
    
    # Точка 1: Колено (Femur end)
    p1 = [
        start_xyz[0] + l_femur * np.cos(gamma) * np.cos(beta),
        start_xyz[1] + l_femur * np.sin(gamma) * np.cos(beta),
        start_xyz[2] + l_femur * np.sin(beta)
    ]
    # Точка 2: Стопа (Tibia end)
    p2 = [
        p1[0] + l_tibia * np.cos(gamma) * np.cos(beta - alpha),
        p1[1] + l_tibia * np.sin(gamma) * np.cos(beta - alpha),
        p1[2] + l_tibia * np.sin(beta - alpha)
    ]
    return [start_xyz, p1, p2]

# --- 3. ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="FSIN Biomorphic Lab", layout="wide")
st.title("Сфиральный Биоморфный Гексапод")

with st.sidebar:
    st.header("Константы")
    wurf = st.number_input("Золотой вурф", 1.0, 2.0, 1.309, format="%.3f")
    mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный"])
    st.divider()
    st.header("Размеры (мм)")
    body_l = st.slider("Длина корпуса", 100, 300, 217)
    body_w = st.slider("Ширина корпуса", 50, 200, 166)
    r_base = st.slider("Масштаб ног", 20, 100, 50)
    time_step = st.slider("Фаза движения (t)", 0.0, 10.0, 0.0)

# Расчет нейросети
net = SfiralNetwork(2, 6, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
current_state = net.forward(time_step)

tab_3d, tab_an = st.tabs(["🕷️ 3D Модель", "📊 Анализ"])

with tab_3d:
    # Отрисовка робота
    fig = go.Figure()
    
    # Корпус (эллипс)
    t_body = np.linspace(0, 2*np.pi, 50)
    bx = (body_l/2) * np.cos(t_body)
    by = (body_w/2) * np.sin(t_body)
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bx*0, mode='lines', line=dict(color='silver', width=10), name="Корпус"))

    # Крепление ног и их ориентация
    # Правая сторона (0, 1, 2), Левая сторона (3, 4, 5)
    mount_angles = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    
    for i in range(6):
        m_ang = mount_angles[i]
        start_xyz = [ (body_l/2)*np.cos(m_ang), (body_w/2)*np.sin(m_ang), 0 ]
        
        # Данные из нейросети управляют подъемом и вылетом ноги
        lift = current_state[i]
        stretch = current_state[(i+1)%6]
        
        pts = calculate_leg_points(start_xyz, m_ang, lift, stretch, wurf, r_base)
        pts = np.array(pts)
        
        fig.add_trace(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode='lines+markers',
            line=dict(color='gold' if i < 3 else 'orange', width=8),
            marker=dict(size=5, color='black'),
            name=f"Leg {i+1}"
        ))

    fig.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-150, 150])), margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig, use_container_width=True)

with tab_an:
    st.subheader("Состояние нейронной сети")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Вектор текущих состояний узлов:")
        st.dataframe(pd.DataFrame(current_state.reshape(1,-1), columns=[f"Node_{i+1}" for i in range(6)]))
    with col2:
        bal = np.sum(current_state)
        st.metric("Мгновенный баланс сил", f"{bal:.18f}")

st.info("Используйте слайдер 'Фаза движения' в боковой панели, чтобы увидеть, как робот шагает.")
