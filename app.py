import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 1. Логика одиночного сфирального узла
class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

# 2. Топология сети с весами Золотого вурфа
class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309):
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        # Матрица весов нормируется по Золотому вурфу для биоморфной балансировки
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size)) * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            net_input = np.dot(inputs, self.weights[:, i])
            outputs.append(neuron.s_transition(net_input))
        return np.array(outputs)

# 3. Графический интерфейс
st.title("ФСИН: Сеть и Золотой вурф")

col1, col2 = st.columns(2)
with col1:
    input_nodes = st.number_input("Входные сигналы", min_value=1, value=3)
with col2:
    output_nodes = st.number_input("Нейроны слоя", min_value=1, value=4)

base_weight = st.number_input("Базовый множитель весов", value=1.309, format="%.3f")

network = SfiralNetwork(input_nodes, output_nodes, base_weight)

# Генерация многомерного входного сигнала
x_values = np.linspace(-10, 10, 400)
inputs_matrix = np.column_stack([np.sin(x_values + i) for i in range(input_nodes)])

# Вычисление состояний сети
y_outputs = np.array([network.forward(inp) for inp in inputs_matrix])

# Отрисовка интерференции узлов
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(output_nodes):
    ax.plot(x_values, y_outputs[:, i], label=f"Узел {i+1}", alpha=0.8)

ax.axhline(0, color='black', linewidth=0.8)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_title("Свитие сил: балансировка состояний сети")
ax.set_xlabel('Временной такт')
ax.set_ylabel('Амплитуда фазового перехода')
ax.legend()

st.pyplot(fig)
