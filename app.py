import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Ядро нейросети с поддержкой антисимметрии ---
class SfiralNeuron:
    def __init__(self, frequency=1.0, phase_shift=np.pi/2):
        self.frequency = frequency
        self.phase_shift = phase_shift

    def s_transition(self, x):
        """Функция активации на базе зеркальной антисимметрии (S-переход)."""
        forward_loop = np.sin(self.frequency * x + self.phase_shift)
        backward_loop = -np.sin(self.frequency * x - self.phase_shift)
        return np.tanh(forward_loop + backward_loop)

class SfiralNetwork:
    def __init__(self, input_size, layer_size, base_weight=1.309, balanced=True):
        self.neurons = [SfiralNeuron() for _ in range(layer_size)]
        self.balanced = balanced
        
        # Генерация весов
        weights = np.random.uniform(-1, 1, (input_size, layer_size))
        
        if self.balanced and layer_size % 2 == 0:
            # Принцип парной антисимметрии: вторая половина нейронов инвертирует первую
            half = layer_size // 2
            weights[:, half:] = -weights[:, :half] 
            
        self.weights = weights * base_weight

    def forward(self, inputs):
        outputs = []
        for i, neuron in enumerate(self.neurons):
            # Для сбалансированного режима сдвигаем фазы парных нейронов
            if self.balanced and i >= len(self.neurons) // 2:
                neuron.phase_shift += np.pi # Сдвиг в противофазу
            
            net_input = np.dot(inputs, self.weights[:, i])
            outputs.append(neuron.s_transition(net_input))
        return np.array(outputs)

# --- Интерфейс ---
st.set_page_config(page_title="FSIN: Biomorphic Balance", layout="wide")
st.title("Балансировка ФСИН: Режим Биоморфного Резонанса")

st.sidebar.header("Настройки гармонии")
mode = st.sidebar.selectbox("Режим работы", ["Случайный (Дисбаланс)", "Антисимметричный (Резонанс)"])
in_nodes = st.sidebar.number_input("Входные каналы", 1, 10, 2)
# Для резонанса лучше четное число нейронов
out_nodes = st.sidebar.number_input("Нейроны слоя (четное число)", 2, 40, 8, step=2)
wurf = st.sidebar.number_input("Золотой вурф", value=1.309, format="%.3f")

# Инициализация сети
is_balanced = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(in_nodes, out_nodes, wurf, balanced=is_balanced)

# Генерация сигнала
x = np.linspace(-10, 10, 500)
inputs = np.column_stack([np.sin(x + i) for i in range(in_nodes)])
y = np.array([net.forward(inp) for inp in inputs])

# Расчет индекса устойчивости в реальном времени
total_sum = np.sum(y, axis=1)
stability_index = np.mean(np.abs(total_sum))

# Визуализация
col_plot, col_stats = st.columns([3, 1])

with col_plot:
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(out_nodes):
        ax.plot(x, y[:, i], alpha=0.6, linewidth=1)
    ax.plot(x, total_sum, color='black', linewidth=2, label='Суммарный вектор (Баланс)')
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax.set_title(f"Состояние сети: {mode}")
    ax.legend()
    st.pyplot(fig)

with col_stats:
    st.metric("Индекс устойчивости", f"{stability_index:.6f}")
    if stability_index < 0.1:
        st.success("Система в резонансе!")
    else:
        st.warning("Требуется балансировка")
    
    st.info(f"Золотой вурф {wurf} обеспечивает структурную норму.")

# Экспорт
df = pd.DataFrame(y, columns=[f"Node_{i+1}" for i in range(out_nodes)])
df.insert(0, "Time", x)
st.download_button("Скачать CSV для Colab", df.to_csv(index=False).encode('utf-8'), "fsin_balanced.csv", "text/csv")
