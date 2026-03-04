import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ядро ФСИН ---
class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        """Функция активации на базе зеркальной антисимметрии."""
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309):
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        # Нормирование весов по Золотому вурфу
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size)) * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            net_input = np.dot(inputs, self.weights[:, i])
            outputs.append(neuron.s_transition(net_input))
        return np.array(outputs)

# --- Интерфейс ---
st.set_page_config(page_title="FSIN Simulator", layout="wide")
st.title("Симулятор фрактальных сфиральных нейронов")

st.markdown("""
Система моделирует динамику сфиральных узлов. В качестве функции активации используется 
**S-образный фазовый переход**, обеспечивающий баланс сил и предотвращающий перегрузку сети.
""")

# Сайдбар с параметрами
st.sidebar.header("Параметры симуляции")
input_nodes = st.sidebar.number_input("Входные каналы", 1, 10, 3)
output_nodes = st.sidebar.number_input("Нейроны слоя", 1, 30, 8)
base_weight = st.sidebar.number_input("Множитель (Вурф)", value=1.309, format="%.3f")

# Вычисления
network = SfiralNetwork(input_nodes, output_nodes, base_weight)
x_values = np.linspace(-10, 10, 400)
inputs_matrix = np.column_stack([np.sin(x_values + i) for i in range(input_nodes)])
y_outputs = np.array([network.forward(inp) for inp in inputs_matrix])

# Визуализация
fig, ax = plt.subplots(figsize=(12, 5))
for i in range(output_nodes):
    ax.plot(x_values, y_outputs[:, i], label=f"Узел {i+1}", alpha=0.7, linewidth=1)

ax.axhline(0, color='black', linewidth=1, alpha=0.5)
ax.set_title("Интерференция фазовых состояний сети")
ax.set_xlabel("Время / Фаза")
ax.set_ylabel("Амплитуда самокомпенсации")
st.pyplot(fig)

# Секция экспорта
df = pd.DataFrame(y_outputs, columns=[f"Node_{i+1}" for i in range(output_nodes)])
df.insert(0, "Phase_Step", x_values)

st.subheader("Генерация данных")
st.dataframe(df.head(10))

csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Скачать матрицу состояний (CSV)",
    data=csv_data,
    file_name='fsin_states_matrix.csv',
    mime='text/csv'
)
