import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.neuron import SfiralNeuron

st.title("ФСИН: Визуализация S-перехода")

frequency = st.slider("Частота витка", 0.1, 5.0, 1.0)
phase = st.slider("Фазовый сдвиг", 0.0, np.pi, np.pi/2)

neuron = SfiralNeuron(frequency=frequency, phase_shift=phase)
x = np.linspace(-10, 10, 400)
y = [neuron.s_transition(val) for val in x]

fig, ax = plt.subplots()
ax.plot(x, y, label="S-образный переход (Антисимметрия)")
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.legend()

st.pyplot(fig)
