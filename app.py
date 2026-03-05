import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- 1. ВЫЧИСЛИТЕЛЬНОЕ ЯДРО (ФСИН) ---
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
        # Генерируем веса один раз
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

# --- 2. ГЕОМЕТРИЯ РОБОТА-НАСЕКОМОГО ---
def get_hexapod_model(wurf, states, body_l, body_w, r_coil):
    # Длины сегментов по Вурфу
    l1 = r_coil * 0.5       # Coxa (Тазовый узел)
    l2 = l1 * wurf          # Femur (Бедро)
    l3 = l2 * wurf          # Tibia (Голень)
    
    traces = []
    # 1. Корпус (Центральная платформа)
    bx = [body_l/2, body_l/2, -body_l/2, -body_l/2, body_l/2]
    by = [body_w/2, -body_w/2, -body_w/2, body_w/2, body_w/2]
    bz = [0, 0, 0, 0, 0]
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='silver', width=10), name="Корпус"))

    # 2. Ноги (6 конечностей)
    # Позиции крепления ног на корпусе
    mount_points = [
        [body_l/2, body_w/2],   # Передняя правая
        [0, body_w/2],          # Средняя правая
        [-body_l/2, body_w/2],  # Задняя правая
        [body_l/2, -body_w/2],  # Передняя левая
        [0, -body_w/2],         # Средняя левая
        [-body_l/2, -body_w/2]  # Задняя левая
    ]
    
    # Базовые углы разворота ног (от оси Y)
    base_angles = [np.pi/4, 0, -np.pi/4, 3*np.pi/4, np.pi, 5*np.pi/4]
    
    for i in range(6):
        start_pos = mount_points[i]
        # Берем данные из нейросети (последнее состояние) для анимации
        # Если индекс устойчивости 0, смещение будет 0
        move_offset = states[-1, i % states.shape[1]] * 0.5 
        
        angle = base_angles[i] + move_offset
        
        # Точка 1: Конец Coxa (горизонтально)
        p1 = [
            start_pos[0] + l1 * np.cos(angle),
            start_pos[1] + l1 * np.sin(angle),
            0
        ]
        # Точка 2: Конец Femur (подъем вверх/наклон)
        p2 = [
            p1[0] + l2 * np.cos(angle),
            p1[1] + l2 * np.sin(angle),
            l1 * 0.5 # Подъем колена
        ]
        # Точка 3: Конец Tibia (опускание к земле)
        p3 = [
            p2[0] + l3 * np.cos(angle) * 0.5,
            p2[1] + l3 * np.sin(angle) * 0.5,
            -l3 * 0.8 # Касание земли
        ]
        
        lx = [start_pos[0], p1[0], p2[0], p3[0]]
        ly = [start_pos[1], p1[1], p2[1], p3[1]]
        lz = [0, 0, p2[2], p3[2]]
        
        traces.append(go.Scatter3d(
            x=lx, y=ly, z=lz, 
            mode='lines+markers',
            line=dict(color='gold' if i < 3 else 'orange', width=7),
            marker=dict(size=4, color='black'),
            name=f"Leg {i+1}"
        ))
        
    return traces

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(page_title="FSIN Biomorphic Lab", layout="wide")
st.title("Сфиральное моделирование: Робот-Насекомое")

with st.sidebar:
    st.header("Математика (ФСИН)")
    mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Узлы сети", 6, 40, 10, step=2)
    wurf = st.sidebar.number_input("Золотой вурф", 1.0, 2.0, 1.309, format="%.3f")
    
    st.divider()
    st.header("Геометрия (Сажени)")
    # Привязываем размеры корпуса к Казенной сажени (217.6 мм)
    body_l = st.slider("Длина корпуса (L)", 100.0, 300.0, 217.6)
    body_w = body_l / wurf
    r_coil = st.slider("Масштаб ног (R_coil)", 20.0, 100.0, 50.0)

# Вычисления сети
net = SfiralNetwork(2, nodes, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
t_ax = np.linspace(-10, 10, 200)
inp = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
states = np.array([net.forward(i) for i in inp])
stab_idx = np.mean(np.abs(np.sum(states, axis=1)))

tab_an, tab_rob = st.tabs(["📉 Анализ резонанса", "🕷️ 3D Робот"])

with tab_an:
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Осциллограмма узлов")
        fig2d, ax2d = plt.subplots(figsize=(10, 4))
        ax2d.plot(t_ax, states, alpha=0.3)
        ax2d.plot(t_ax, np.sum(states, axis=1), color='black', lw=2, label='Суммарный вектор')
        ax2d.axhline(0, color='red', ls='--')
        st.pyplot(fig2d)
    with c2:
        st.metric("Stability Index", f"{stab_idx:.18f}")
        fig_h, ax_h = plt.subplots(figsize=(5, 5))
        sns.heatmap(pd.DataFrame(states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab_rob:
    st.subheader("Визуализация Гексапода")
    st.info(f"Ширина корпуса (W) автоматически рассчитана по Вурфу: {body_w:.1f} мм")
    
    robot_data = get_hexapod_model(wurf, states, body_l, body_w, r_coil)
    fig_rob = go.Figure(data=robot_data)
    
    # Настройка осей для корректного отображения
    fig_rob.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis=dict(title='X (Длина)', range=[-250, 250]),
            yaxis=dict(title='Y (Ширина)', range=[-250, 250]),
            zaxis=dict(title='Z (Высота)', range=[-150, 150])
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=700
    )
    st.plotly_chart(fig_rob, use_container_width=True)

st.download_button("Экспорт CSV", pd.DataFrame(states).to_csv(index=False).encode('utf-8'), "robot_fsin_data.csv")
