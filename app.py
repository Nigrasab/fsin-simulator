import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Ядро ФСИН (интегрировано в основной файл)
class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

# Графический интерфейс Streamlit
st.title("ФСИН: Визуализация баланса сил")

# Интерактивные ползунки
frequency = st.slider("Частота витка", 0.1, 5.0, 1.2)
phase = st.slider("Фазовый сдвиг (рад)", 0.0, float(np.pi), float(np.pi/2))

# Вычисления
neuron = SfiralNeuron(frequency=frequency, phase_shift=phase)
x_values = np.linspace(-10, 10, 400)
y_values = [neuron.s_transition(val) for val in x_values]

# Отрисовка графика
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x_values, y_values, label="S-образный переход", color='#1f77b4')
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
ax.set_xlabel('Входной сигнал')
ax.set_ylabel('Состояние (самокомпенсация)')
ax.legend()

# Вывод в браузер
st.pyplot(fig)
