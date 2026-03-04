import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- ВЫЧИСЛИТЕЛЬНОЕ ЯДРО (МАТЕМАТИКА) ---
class SfiralNeuron:
    def __init__(self, frequency=1.236, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

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

# --- ГЕОМЕТРИЧЕСКОЕ ЯДРО (ПО ВАШЕМУ СКРИПТУ BLENDER) ---
def generate_correct_sphiral(r_coil, h_coil, h_s, turns, res=1000):
    r_arc = r_coil / 2.0
    right_points = []
    
    # 1. S-Дуга (от 0 до R_coil)
    res_arc = int(res * 0.3)
    for i in range(res_arc + 1):
        t = i / res_arc
        phi = np.pi * (1 - t)
        x = r_arc + r_arc * np.cos(phi)
        y = -r_arc * np.sin(phi)
        z = (h_s / 2) * t
        right_points.append([x, y, z])
        
    # 2. Основной виток (от R_coil по кругу)
    res_coil = int(res * 0.7)
    z_start_coil = h_s / 2
    for i in range(1, res_coil + 1):
        t = i / res_coil
        theta = turns * 2 * np.pi * t
        x = r_coil * np.cos(theta)
        y = r_coil * np.sin(theta)
        z = z_start_coil + (h_coil * t)
        right_points.append([x, y, z])
    
    right_points = np.array(right_points)
    
    # 3. Антисимметрия (P_left = -P_right)
    # Отражаем и инвертируем правую часть
    left_points = -right_points[::-1] # Разворот массива и инверсия знаков
    
    # Соединяем (левая часть переходит в правую через 0,0,0)
    full_coords = np.vstack([left_points, right_points])
    return full_coords

# --- ИНТЕРФЕЙС STREAMLIT ---
st.set_page_config(page_title="FSIN Sphiral Analyzer", layout="wide")
st.title("Комплекс анализа Сфиральной Геометрии")

# Сайдбар управления
st.sidebar.header("Настройки Геометрии S-узла")
r_val = st.sidebar.slider("Радиус витка (R_coil)", 10.0, 100.0, 50.0)
h_coil = st.sidebar.slider("Высота витков (Height_Coil)", 10.0, 100.0, 30.0)
h_s = st.sidebar.slider("Высота S-перехода (Height_S)", 5.0, 50.0, 10.0)
wurf = st.sidebar.number_input("Золотой вурф", value=1.309, format="%.3f")

st.sidebar.markdown("---")
st.sidebar.header("Параметры Нейросети")
mode = st.sidebar.selectbox("Режим", ["Антисимметричный (Резонанс)", "Случайный"])
nodes = st.sidebar.number_input("Нейроны", 2, 40, 10, step=2)

# Расчеты сети
net = SfiralNetwork(2, nodes, wurf, balanced=(mode == "Антисимметричный (Резонанс)"))
time_steps = np.linspace(-10, 10, 400)
inputs = np.column_stack([np.sin(time_steps), np.cos(time_steps)])
states = np.array([net.forward(inp) for inp in inputs])
total_balance = np.sum(states, axis=1)
stability_index = np.mean(np.abs(total_balance))

# Визуализация 2D
col1, col2 = st.columns([2, 1])
with col1:
    fig_2d, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_steps, states, alpha=0.3)
    ax.plot(time_steps, total_balance, color='black', linewidth=2, label='Баланс')
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_title("Фазовый баланс ФСИН")
    st.pyplot(fig_2d)
with col2:
    st.metric("Индекс устойчивости", f"{stability_index:.6f}")
    if stability_index < 0.1:
        st.success("РЕЗОНАНС ДОСТИГНУТ")
    else:
        st.warning("ДИСБАЛАНС")

# --- 3D ВИЗУАЛИЗАЦИЯ (ПО СКРИПТУ BLENDER) ---
st.subheader("Топология Сфирали (S-элемент)")
coords = generate_correct_sphiral(r_val, h_coil, h_s, turns=1.0)

fig_3d = go.Figure(data=[go.Scatter3d(
    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
    mode='lines',
    line=dict(color='red', width=6),
    name="Sphiral Path"
)])

# Добавляем Красную Линию (Ось) как в Blender
fig_3d.add_trace(go.Scatter3d(
    x=[-r_val*1.5, r_val*1.5], y=[0, 0], z=[0, 0],
    mode='lines',
    line=dict(color='white', width=2, dash='dash'),
    name="Reference Axis"
))

fig_3d.update_layout(scene=dict(aspectmode='data'), margin=dict(l=0, r=0, b=0, t=0))
st.plotly_chart(fig_3d, use_container_width=True)

# Экспорт
df = pd.DataFrame(states, columns=[f"Node_{i+1}" for i in range(nodes)])
st.download_button("Скачать CSV для анализа", df.to_csv(index=False).encode('utf-8'), "fsin_sphiral.csv")
