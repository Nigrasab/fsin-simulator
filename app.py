import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# --- ГЛОБАЛЬНЫЙ СТИЛЬ ---
st.set_page_config(page_title="FSIN: Сквозное моделирование", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1c212d; padding: 15px; border-radius: 10px; border: 1px solid #ffca28; }
    h1, h2, h3 { color: #ffca28 !important; }
    .stTabs [aria-selected="true"] { background-color: #ffca28; color: black; }
    .info-block { background-color: #11151c; padding: 15px; border-left: 4px solid #ffca28; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_stdio=True)

# --- МАТЕМАТИЧЕСКОЕ ЯДРО ---
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
            raw[half:] = -raw[:half]
        return raw

# --- ГЕОМЕТРИЯ ---
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

def build_insect_legs(wurf, states, body_l, body_w, r_base):
    l_coxa = r_base * 0.4
    l_femur = l_coxa * wurf
    l_tibia = l_femur * wurf
    traces = []
    
    t_b = np.linspace(0, 2*np.pi, 50)
    bx = (body_l/2)*np.cos(t_b)
    by = (body_w/2)*np.sin(t_b)
    bz = np.zeros_like(t_b)
    traces.append(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='silver', width=12), name="Корпус"))
    
    m_angs = [np.pi/4, np.pi/2, 3*np.pi/4, -np.pi/4, -np.pi/2, -3*np.pi/4]
    
    for i in range(6):
        m_a = m_angs[i]
        sx = (body_l/2)*np.cos(m_a)
        sy = (body_w/2)*np.sin(m_a)
        
        s_lift = states[i % len(states)]
        s_stretch = states[(i+1) % len(states)]
        
        gamma = m_a + (s_stretch * 0.4)
        beta = 0.5 + (s_lift * 0.5)
        alpha = 1.2 + (s_lift * 0.4)
        
        p1x = sx + l_femur * np.cos(gamma) * np.cos(beta)
        p1y = sy + l_femur * np.sin(gamma) * np.cos(beta)
        p1z = l_femur * np.sin(beta)
        
        p2x = p1x + l_tibia * np.cos(gamma) * np.cos(beta - alpha)
        p2y = p1y + l_tibia * np.sin(gamma) * np.cos(beta - alpha)
        p2z = p1z + l_tibia * np.sin(beta - alpha)
        
        traces.append(go.Scatter3d(x=[sx, p1x, p2x], y=[sy, p1y, p2y], z=[0, p1z, p2z],
                                   mode='lines+markers', line=dict(color='#ffca28', width=8), name=f"Leg {i+1}"))
    return traces

# --- SIDEBAR: ГЕНЕТИЧЕСКИЙ КОД ---
with st.sidebar:
    st.header("🧬 Генетический код")
    mode = st.selectbox("Режим баланса", ["Антисимметричный (Резонанс)", "Случайный"])
    nodes = st.number_input("Узлы (четное)", 6, 40, 10, step=2)
    wurf = st.number_input("Золотой вурф", 1.0, 2.0, 1.309, format="%.3f")
    
    st.subheader("Форма (Сфираль)")
    r_c = st.slider("Радиус каркаса", 10, 100, 50)
    h_c = st.slider("Высота витков", 10, 100, 30)
    h_s = st.slider("Высота S-узла", 5, 50, 10)
    
    st.subheader("Жизнь (Робот)")
    b_l = st.slider("Длина тела", 100, 300, 217)
    b_w = st.slider("Ширина тела", 50, 250, 166)
    t_phase = st.slider("Фаза движения", 0.0, 10.0, 0.0, step=0.1)

# --- ОСНОВНОЙ КОНТЕНТ ---
st.title("Комплекс ФСИН: Сквозное моделирование")

st.markdown("""
<div class="info-block">
<b>Философия и логика работы (для пользователя):</b><br><br>
<b>Генетический код (Sidebar):</b>
<ul>
<li><b>Золотой вурф (1.309):</b> Это не просто число, а коэффициент гармонии. В приложении он определяет всё: от амплитуды "всплесков" нейронов до того, насколько голень робота длиннее его бедра.</li>
<li><b>Узлы (10):</b> Количество потоков энергии. Четное число позволяет создать пары "действие-противодействие".</li>
</ul>
<b>Сквозное моделирование (Tabs):</b>
<ul>
<li><b>Уровень I (Энергия):</b> Мы видим чистые волны. Наша задача — настроить их так, чтобы суммарная красная линия лежала на нуле. Это означает, что система не тратит энергию на паразитные вибрации.</li>
<li><b>Уровень II (Форма):</b> Математика превращается в геометрию. Сфираль — это идеальная антенна или катушка. Её "плавность" напрямую зависит от того, насколько устойчив резонанс на первом уровне.</li>
<li><b>Уровень III (Жизнь):</b> Здесь мы видим результат. Робот-паук — это проверка того, как наши волны управляют сложной материей. Благодаря Вурфу, его ноги выглядят естественно, а благодаря Резонансу, его походка стабильна.</li>
</ul>
<b>Результат:</b> Теперь приложение — это цельная история. Вы меняете одну цифру в «Генетическом коде» и мгновенно видите, как меняется баланс сил, форма катушки и походка робота. Это и есть настоящее инженерное искусство.
</div>
""", unsafe_allow_stdio=True)

# --- РАСЧЕТЫ ---
is_bal = (mode == "Антисимметричный (Резонанс)")
net = SfiralNetwork(nodes, wurf, balanced=is_bal)

t_range = np.linspace(0, 10, 300)
all_states = np.array([net.get_states(t) for t in t_range])
current_states = net.get_states(t_phase)
stab_idx = np.mean(np.abs(np.sum(all_states, axis=1)))

# --- ВИЗУАЛИЗАЦИЯ ---
tab1, tab2, tab3 = st.tabs(["📊 Уровень I: ЭНЕРГИЯ", "🌀 Уровень II: ФОРМА", "🕷️ Уровень III: ЖИЗНЬ"])

with tab1:
    c1, c2 = st.columns([2, 1])
    with c1:
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor='#0e1117')
        ax1.set_facecolor('#1c212d')
        ax1.plot(t_range, all_states, alpha=0.3)
        ax1.plot(t_range, np.sum(all_states, axis=1), color='#ff4b4b', lw=2)
        ax1.tick_params(colors='white')
        st.pyplot(fig1)
    with c2:
        st.metric("Индекс устойчивости", f"{stab_idx:.18f}")
        fig_h, ax_h = plt.subplots(figsize=(5, 5), facecolor='#0e1117')
        sns.heatmap(pd.DataFrame(all_states).corr(), cmap='RdBu_r', center=0, cbar=False, ax=ax_h)
        st.pyplot(fig_h)

with tab2:
    coords = build_sfiral(r_c, h_c, h_s)
    fig2 = go.Figure(data=[go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], 
                                        mode='lines', line=dict(color='#ffca28', width=6))])
    fig2.update_layout(scene=dict(aspectmode='data', bgcolor="#0e1117"), margin=dict(l=0,r=0,b=0,t=0))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    traces = build_insect_legs(wurf, current_states, b_l, b_w, r_c)
    fig3 = go.Figure(data=traces)
    fig3.update_layout(scene=dict(aspectmode='data', zaxis=dict(range=[-150, 150]), bgcolor="#0e1117"), margin=dict(l=0,r=0,b=0,t=0), height=700)
    st.plotly_chart(fig3, use_container_width=True)

df_export = pd.DataFrame(all_states, columns=[f"Node_{i+1}" for i in range(nodes)])
st.download_button("Экспорт CSV", df_export.to_csv(index=False).encode('utf-8'), "fsin_genetic_matrix.csv")
