import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309):
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size)) * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            net_input = np.dot(inputs, self.weights[:, i])
            outputs.append(neuron.s_transition(net_input))
        return np.array(outputs)

st.title("ФСИН: Сеть, Золотой вурф и Экспорт")

col1, col2 = st.columns(2)
with col1:
    input_nodes = st.number_input("Входные сигналы", min_value=1, value=3)
with col2:
    output_nodes = st.number_input("Нейроны слоя", min_value=1, value=4)

base_weight = st.number_input("Базовый множитель весов", value=1.309, format="%.3f")

network = SfiralNetwork(input_nodes, output_nodes, base_weight)

x_values = np.linspace(-10, 10, 400)
inputs_matrix = np.column_stack([np.sin(x_values + i) for i in range(input_nodes)])
y_outputs = np.array([network.forward(inp) for inp in inputs_matrix])

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

# Блок формирования и экспорта датасета
st.subheader("Матрица вычисленных состояний")

# Сборка данных в таблицу
df = pd.DataFrame(y_outputs, columns=[f"Узел_{i+1}" for i in range(output_nodes)])
df.insert(0, "Временной_такт", x_values)

# Вывод первых 5 строк для визуального контроля
st.dataframe(df.head())

# Конвертация и кнопка скачивания
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Скачать матрицу (CSV)",
    data=csv_data,
    file_name='fsin_states_matrix.csv',
    mime='text/csv',
)
