import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- Вычислительное ядро ФСИН ---
class SfiralNeuron:
    def __init__(self, frequency=1.236, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        """S-образный фазовый переход на базе зеркальной антисимметрии."""
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        
        # Инициализация весовых коэффициентов
        weights = np.random.uniform(-1, 1, (input_size, layer_size))
        
        if self.balanced and layer_size % 2 == 0:
            # Принцип парного свития: инверсия весов для второй половины слоя
            half = layer_size // 2
            weights[:, half:] = -weights[:, :half]
            
        self.weights = weights * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            # В режиме резонанса сдвигаем фазу парных нейронов на PI (противофаза)
            p_offset = np.pi if (self.balanced and i >= self.layer_size // 2) else 0
            
            # Локальная копия параметров для расчета
            current_phase = neuron.phase_shift + p_offset
            
            net_input = np.dot(inputs, self.weights[:, i])
            
            # Расчет S-перехода
            f_loop = np.sin(neuron.frequency * net_input + current_phase)
            b_loop = -np.sin(neuron.frequency * net_input - current_phase)
            val = np.tanh(f_loop + b_loop)
            outputs.append(val)
        return np.array(outputs)

# --- Интерфейс управления ---
st.set_page_config(page_title="FSIN Simulator 2.0", layout="wide")
st.title("Научно-исследовательский комплекс ФСИН")

st.markdown("""
Симулятор моделирует динамику нейронов на принципах зеркальной антисимметрии. 
Цель: достижение **Биоморфного Резонанса** (Stability Index < 0.1) при использовании **Золотого вурфа (1.309)**.
""")

# Сайдбар параметров
st.sidebar.header("Параметры системы")
mode = st.sidebar.selectbox("Режим балансировки", ["Антисимметричный (Резонанс)", "Случайный (Дисбаланс)"])
in_nodes = st.sidebar.number_input("Входные каналы", 1, 10, 2)
out_nodes = st.sidebar.number_input("Нейроны слоя (четное)", 2, 40, 10, step=2)
wurf = st.sidebar.number_input("Золотой вурф", value=1.309, format="%.3f")
freq = st.sidebar.slider("Базовая частота", 0.1, 5.0, 1.236)

# Вычисления
is_balanced = (mode == "Антисимметричный (Резонанс)")
network = SfiralNetwork(in_nodes, out_nodes, wurf, balanced=is_balanced)

# Генерация сигналов
time_steps = np.linspace(-10, 10, 500)
input_signals = np.column_stack([np.sin(time_steps + i) for i in range(in_nodes)])
node_states = np.array([network.forward(inp) for inp in input_signals])

# Расчет метрик
total_balance_vector = np.sum(node_states, axis=1)
stability_index = np.mean(np.abs(total_balance_vector))

# --- Визуализация данных ---
col_2d, col_stats = st.columns([3, 1])

with col_2d:
    st.subheader("Динамика фазовых состояний")
    fig_2d, ax = plt.subplots(figsize=(10, 5))
    for i in range(out_nodes):
        ax.plot(time_steps, node_states[:, i], alpha=0.5, linewidth=1)
    
    # Суммарный вектор (черная линия)
    ax.plot(time_steps, total_balance_vector, color='black', linewidth=2.5, label='Суммарный баланс')
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.4)
    ax.set_xlabel("Время / Фаза")
    ax.set_ylabel("Амплитуда S-перехода")
    ax.legend()
    st.pyplot(fig_2d)

with col_stats:
    st.subheader("Метрики резонанса")
    st.metric("Индекс устойчивости", f"{stability_index:.6f}")
    
    if stability_index < 0.1:
        st.success("СТАТУС: БИОМОРФНЫЙ РЕЗОНАНС")
        st.info("Система самокомпенсирована. Внутренние напряжения отсутствуют.")
    else:
        st.warning("СТАТУС: ДИСФАЗИЯ")
        st.error("Система перегружена. Требуется балансировка пар.")

# --- 3D Геометрия сфирали ---
st.subheader("Геометрическая проекция резонансного узла")
t_geo = np.linspace(0, 10 * np.pi, 500)
# Высота спирали прямо зависит от Золотого вурфа
z_geo = np.linspace(0, 10 * wurf, 500) 
x_geo = np.cos(t_geo) * (1 + 0.2 * np.sin(t_geo/out_nodes))
y_geo = np.sin(t_geo) * (1 + 0.2 * np.sin(t_geo/out_nodes))

fig_3d = go.Figure(data=[go.Scatter3d(
    x=x_geo, y=y_geo, z=z_geo,
    mode='lines',
    line=dict(color='gold', width=5)
)])

fig_3d.update_layout(
    scene=dict(
        xaxis_title='X (Антисимметрия)',
        yaxis_title='Y (Синхронность)',
        zaxis_title='Z (Поток / Вурф)'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
st.plotly_chart(fig_3d, use_container_width=True)

# --- Экспорт ---
df_export = pd.DataFrame(node_states, columns=[f"Node_{i+1}" for i in range(out_nodes)])
df_export.insert(0, "Phase_Step", time_steps)

st.subheader("Инструменты экспорта")
st.download_button(
    label="Скачать матрицу состояний (CSV)",
    data=df_export.to_csv(index=False).encode('utf-8'),
    file_name='fsin_states_matrix.csv',
    mime='text/csv'
)
