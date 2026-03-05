import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- ГЛОБАЛЬНАЯ СТИЛИЗАЦИЯ (Modern Dark) ---
st.set_page_config(page_title="FSIN: Сквозное моделирование", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c212d; padding: 15px; border-radius: 10px; border: 1px solid #ffca28; }
    h1, h2, h3 { color: #ffca28 !important; font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #11151c; border-radius: 4px 4px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #ffca28; color: black; }
    </style>
    """, unsafe_allow_stdio=True)

# --- МАТЕМАТИЧЕСКИЙ ГЕНОМ ---
class SfiralNetwork:
    def __init__(self, layer_size, base_weight=1.309, balanced=True):
        self.layer_size = layer_size
        self.balanced = balanced
        np.random.seed(42)
        self.weights = np.random.uniform(-0.5, 0.5, (2, layer_size)) * base_weight

    def get_states(self, t_val):
        x, y = np.sin(1.236 * t_val), np.cos(1.236 * t_val)
        raw = np.tanh(np.dot([x, y], self.weights))
        if self.balanced:
            half = self.layer_size // 2
            raw[half:] = -raw[:half] # Принудительная антисимметрия
        return raw

# --- ГЕОМЕТРИЧЕСКИЙ ПРОЕКТОР ---
def build_sfiral(r, hc, hs):
    res = 600
    r_a, pts = r / 2.0, []
    for i in range(int(res*0.3)+1):
        t = i / (res*0.3); phi = np.pi * (1-t)
        pts.append([r_a + r_a*np.cos(phi), -r_a*np.sin(phi), (hs/2)*t])
    for i in range(1, int(res*0.7)+1):
        t = i / (res*0.7)
        pts.append([r*np.cos(2*np.pi*t), r*np.sin(2*np.pi*t), (hs/2) + hc*t])
    pts = np.array(pts)
    return np.vstack([-pts[::-1], pts])

# --- БИОМОРФНАЯ КИНЕМАТИКА ---
def build_insect(wurf, current_states, l, w, r_base):
    l_coxa = r_base * 0.3
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    traces = []
    # Тело
    t_b = np.linspace(0, 2*np.pi, 50)
    traces.append(go.Scatter3d(x=(l/2)*np.cos(t_b), y=(w/2)*np.sin(t_b), z=t_b*0, 
                               mode='lines', line=dict(color='silver', width=12), name="Корпус"))
    # Ноги
    m_angs = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    for i in range(6):
        m_a = m_angs[i]
        start = [(l/2)*np.cos(m_a), (w/2)*np.sin(m_a), 0]
        s = current_states[i % len(current_states)]
        ang = m_a + (s * 0.2)
        p1 = [start[0] + l_femur*np.cos(ang), start[1] + l_femur*np.sin(ang), 20*s]
        p2 = [p1[0] + l_tibia*np.cos(ang), p1[1] + l_tibia*np.sin(ang), -50]
        traces.append(go.Scatter3d(x=[start[0], p1[0], p2[0]], y=[start[1], p1[1], p2[1]], z=[0, p1[2], p2[2]],
                                   mode='lines+markers', line=dict(color='#ffca28', width=8), name=f"Leg {i+1}"))
    return traces

# --- SIDEBAR: ГЕНЕТИЧЕСКИЙ КОД ---
with st.sidebar:
    st.header("🧬 ГЕНЕТИЧЕСКИЙ КОД")
    st.markdown("""
    **Панель Генетического Кода** — это фундамент системы. Изменения здесь пронизывают все уровни: 
    от микро-вибрации нейрона до длины лапы робота. Это воплощение **Сквозного моделирования**.
    """)
    with st.expander("📡 Резонансные константы", expanded=True):
        mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный (Хаос)"])
        nodes = st.number_input("Нейроны (Узлы)", 6, 40, 10, step=2)
        wurf = st.number_input("Золотой вурф (Масштаб)", 1.0, 2.0, 1.309, format="%.3f")
    with st.expander("🌀 Параметры Материи (Сфираль)", expanded=True):
        r_c = st.slider("Радиус каркаса", 10, 100, 50)
        h_c = st.slider("Высота витков", 10, 100, 30)
        h_s = st.slider("Высота S-узла", 5, 50, 10)
    with st.expander("🕷️ Параметры Воплощения (Робот)", expanded=True):
        b_l = st.slider("Длина тела", 100, 300, 217)
        b_w = st.slider("Ширина тела", 50, 250, 166)
        t_phase = st.slider("Фаза движения", 0.0, 10.0, 0.0)

# --- ОСНОВНОЙ КОНТЕНТ ---
st.title("Комплекс ФСИН: Сквозное моделирование")
st.markdown("""
**Логика системы:** Путь реализации идеи от **Энергии (Сигнал)** через **Форму (Сфираль)** к **Жизни (Робот).** Все три уровня неразрывно связаны коэффициентом Золотого вурфа ($1.309$) и принципом антисимметричного резонанса.
""")

# Инициализация
is_bal = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(nodes, wurf, balanced=is_bal)
t_range = np.linspace(0, 10, 300)
all_states = np.array([net.get_states(t) for t in t_range])
current_states = net.get_states(t_phase)
stab_idx = np.mean(np.abs(np.sum(all_states, axis=1)))

tab1, tab2, tab3 = st.tabs(["📊 Уровень I: ЭНЕРГИЯ", "🌀 Уровень II: ФОРМА", "🕷️ Уровень III: ЖИЗНЬ"])

with tab1:
    st.subheader("Математика Антисимметрии")
    st.markdown("Здесь мы настраиваем чистоту волны. Цель — свести суммарный вектор (красный) к нулю.")
    c1, c2 = st.columns([2, 1])
    with c1:
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
        ax1.set_facecolor('#1c212d')
        ax1.plot(t_range, all_states, alpha=0.3, lw=1)
        ax1.plot(t_range, np.sum(all_states, axis=1), color='#ff4b4b', lw=2, label='Баланс')
        ax1.tick_params(colors='white'); st.pyplot(fig1)
    with c2:
        st.metric("Индекс устойчивости", f"{stab_idx:.18f}")
        fig_h, ax_h = plt.subplots(figsize=(5, 5), facecolor='#0e1117')
        sns.heatmap(pd.DataFrame(all_states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab2:
    st.subheader("Геометрия Сфирального Узла")
    st.markdown("Проекция сигналов в структуру. Форма сфирали показывает путь тока, который не встречает сопротивления.")
    coords = build_sfiral(r_c, h_c, h_s)
    fig2 = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], 
                                        mode='lines', line=dict(color='#ffca28', width=6))])
    fig2.update_layout(scene=dict(aspectmode='data', bgcolor="#0e1117"), margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Биоморфный Гексапод")
    st.markdown("Робот управляется импульсами ФСИН. Пропорции ног и походка напрямую зависят от Золотого вурфа.")
    robot_traces = build_insect(wurf, current_states, b_l, b_w, r_c)
    fig3 = go.Figure(data=robot_traces)
    fig3.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-100, 100]), bgcolor="#0e1117"), 
                       margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig3, use_container_width=True)

st.download_button("💾 Экспорт Генетической Матрицы", pd.DataFrame(all_states).to_csv().encode('utf-8'), "fsin_genetic_matrix.csv")
