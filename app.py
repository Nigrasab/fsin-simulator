import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- СТИЛИЗАЦИЯ ИНТЕРФЕЙСА ---
st.set_page_config(page_title="FSIN: End-to-End Modeling", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #e0e0e0; }
    .stMetric { background-color: #1c212d; padding: 15px; border-radius: 10px; border: 1px solid #ff4b4b; }
    .stSidebar { background-color: #11151c !important; }
    h1, h2, h3 { color: #ffca28 !important; }
    </style>
    """, unsafe_allow_stdio=True)

# --- 1. МАТЕМАТИЧЕСКОЕ ЯДРО (ФСИН) ---
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
        np.random.seed(42)
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

# --- 2. ГЕОМЕТРИЯ (БИОМОРФНЫЙ РОБОТ И СФИРАЛЬ) ---
def generate_sfiral_coords(r, hc, hs, res=800):
    r_a, pts = r / 2.0, []
    res_arc = int(res * 0.3)
    for i in range(res_arc + 1):
        t = i / res_arc
        phi = np.pi * (1 - t)
        pts.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    res_coil = int(res * 0.7)
    z_start = hs / 2
    for i in range(1, res_coil + 1):
        t = i / res_coil
        pts.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), z_start + hc*t])
    pts = np.array(pts)
    return np.vstack([-pts[::-1], pts])

def calculate_leg_points(start_xyz, base_angle, lift_val, stretch_val, wurf, r_base):
    # Кинематика: Длины сегментов связаны через Вурф
    l_coxa = r_base * 0.3
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    gamma = base_angle + (stretch_val * 0.15)
    beta = 0.4 + (lift_val * 0.35)
    alpha = 1.1 + (lift_val * 0.25)
    p1 = [start_xyz[0] + l_femur*np.cos(gamma)*np.cos(beta),
          start_xyz[1] + l_femur*np.sin(gamma)*np.cos(beta),
          start_xyz[2] + l_femur*np.sin(beta)]
    p2 = [p1[0] + l_tibia*np.cos(gamma)*np.cos(beta-alpha),
          p1[1] + l_tibia*np.sin(gamma)*np.cos(beta-alpha),
          p1[2] + l_tibia*np.sin(beta-alpha)]
    return [start_xyz, p1, p2]

# --- 3. ПАНЕЛЬ "ГЕНЕТИЧЕСКИЙ КОД" (SIDEBAR) ---
with st.sidebar:
    st.title("🧬 Генетический код")
    st.info("Это фундамент системы. Изменения здесь пронизывают все уровни: от вибрации нейрона до длины лапы робота.")
    
    with st.expander("📡 Резонансные константы", expanded=True):
        mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный (Хаос)"])
        nodes = st.number_input("Узлы сети (Нейроны)", 6, 40, 10, step=2)
        wurf = st.number_input("Золотой вурф (Масштаб)", 1.0, 2.0, 1.309, format="%.3f")
    
    with st.expander("🌀 Параметры Материи (Сфираль)", expanded=True):
        r_c = st.slider("Радиус каркаса", 10, 100, 50)
        h_c = st.slider("Высота витков", 10, 100, 30)
        h_s = st.slider("Высота S-узла", 5, 50, 10)
    
    with st.expander("🕷️ Параметры Воплощения (Робот)", expanded=True):
        body_l = st.slider("Длина тела", 100, 300, 217)
        body_w = st.slider("Ширина тела", 50, 250, 166)
        t_phase = st.slider("Фаза движения", 0.0, 10.0, 0.0)

# --- 4. ОСНОВНОЙ ЭКРАН ---
st.header("Сквозное моделирование: От Сингулярности к Роботу")
st.markdown("""
Это процесс бесшовной трансляции математики в физику. 
**Уровень 1:** Чистая энергия (Графики) → **Уровень 2:** Структура узла (Сфираль) → **Уровень 3:** Кинематика жизни (Робот).
""")

# Расчеты
is_bal = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(2, nodes, wurf, balanced=is_bal)
t_ax = np.linspace(-10, 10, 400)
signals = np.column_stack([np.sin(t_ax), np.cos(t_ax)])
states = np.array([net.forward(s) for s in signals])
stab_idx = np.mean(np.abs(np.sum(states, axis=1)))

# ВКЛАДКИ
tab1, tab2, tab3 = st.tabs(["📊 Уровень I: ЭНЕРГИЯ", "🌀 Уровень II: ФОРМА", "🕷️ Уровень III: ЖИЗНЬ"])

with tab1:
    st.subheader("Математика Антисимметрии")
    c_a, c_b = st.columns([2, 1])
    with c_a:
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
        ax1.set_facecolor('#1c212d')
        ax1.plot(t_ax, states, alpha=0.3, lw=1)
        ax1.plot(t_ax, np.sum(states, axis=1), color='#ff4b4b', lw=2.5, label='Суммарный резонанс')
        ax1.tick_params(colors='white')
        st.pyplot(fig1)
    with c_b:
        st.metric("Индекс устойчивости", f"{stab_idx:.18f}")
        st.markdown("**Термокарта:** Показывает, насколько идеально левая сторона гасит правую.")
        fig_h, ax_h = plt.subplots(figsize=(5, 5), facecolor='#0e1117')
        sns.heatmap(pd.DataFrame(states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab2:
    st.subheader("Геометрия Сфирального Узла")
    st.markdown("Это проекция ваших сигналов в 3D пространство. Идеальное свитие без углов — признак чистого резонанса.")
    coords = generate_sfiral_coords(r_c, h_c, h_s)
    fig3 = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], 
                                         mode='lines', line=dict(color='#ffca28', width=6))])
    fig3.update_layout(scene=dict(aspectmode='data', bgcolor="#0e1117"), margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("Биоморфный Гексапод")
    st.markdown("Робот использует сигналы нейросети как импульсы для мышц. При индексе устойчивости = 0 движение максимально плавно.")
    curr_state = net.forward(np.array([np.sin(t_phase), np.cos(t_phase)]))
    fig_r = go.Figure()
    # Корпус
    t_b = np.linspace(0, 2*np.pi, 50)
    fig_r.add_trace(go.Scatter3d(x=(body_l/2)*np.cos(t_b), y=(body_w/2)*np.sin(t_b), z=t_b*0, 
                                 mode='lines', line=dict(color='silver', width=10), name="Корпус"))
    # Ноги
    m_angles = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    for i in range(6):
        m_ang = m_angles[i]
        start = [(body_l/2)*np.cos(m_ang), (body_w/2)*np.sin(m_ang), 0]
        pts = np.array(calculate_leg_points(start, m_ang, curr_state[i%nodes], curr_state[(i+1)%nodes], wurf, r_c))
        fig_r.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='lines+markers',
                                     line=dict(color='#ffca28' if i<3 else '#ff9100', width=8), name=f"Leg {i+1}"))
    fig_r.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-150, 150])), margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig_r, use_container_width=True)

st.download_button("💾 Экспорт Генетической Матрицы (CSV)", pd.DataFrame(states).to_csv(index=False).encode('utf-8'), "fsin_genetic_matrix.csv")
