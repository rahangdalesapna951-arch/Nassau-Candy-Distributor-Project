"""
Nassau Candy Distributor
Factory Reallocation & Shipping Optimization Dashboard
Run: streamlit run nassau_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy | Factory Optimizer",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --green-dark:  #1B4332;
    --green-mid:   #2D6A4F;
    --green-main:  #40916C;
    --green-light: #52B788;
    --green-pale:  #95D5B2;
    --cream:       #F8F4EE;
    --gold:        #C9A84C;
    --red:         #C0392B;
    --bg:          #0F1C14;
    --card:        #162519;
    --border:      #2D6A4F44;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: #E8F5E9;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #1a3a28 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, #40916C33, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    color: #fff;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-sub {
    color: var(--green-pale);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 0.05em;
}

/* KPI cards */
.kpi-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); border-color: var(--green-light); }
.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--green-pale);
    line-height: 1;
}
.kpi-label {
    font-size: 0.75rem;
    color: #78A98A;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.kpi-delta { font-size: 0.8rem; color: var(--gold); margin-top: 0.2rem; }

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #fff;
    border-left: 4px solid var(--green-light);
    padding-left: 0.75rem;
    margin: 1.5rem 0 1rem 0;
}

/* Recommendation cards */
.rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: var(--green-light); }
.rec-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 900;
    color: var(--green-pale);
    min-width: 2rem;
}
.rec-body { flex: 1; }
.rec-product { font-weight: 600; color: #fff; font-size: 0.95rem; }
.rec-detail { font-size: 0.8rem; color: #78A98A; margin-top: 0.2rem; }
.rec-badge {
    background: var(--green-mid);
    color: #fff;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}
.badge-risk {
    background: #7b1a13;
    color: #fca5a5;
}
.badge-safe {
    background: #1a4731;
    color: #86efac;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0a1510;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .css-1d391kg { padding: 1.5rem 1rem; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.3rem;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #78A98A;
    font-weight: 500;
    font-size: 0.85rem;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: var(--green-mid) !important;
    color: #fff !important;
}

/* Slider and select */
.stSelectbox > div, .stMultiSelect > div { background: var(--card) !important; }

/* Metric override */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE DATA
# ─────────────────────────────────────────────────────────────────────────────
FACTORIES = {
    "Lot's O' Nuts":     {'lat': 32.881893, 'lon': -111.768036, 'state': 'AZ'},
    "Wicked Choccy's":   {'lat': 32.076176, 'lon': -81.088371,  'state': 'GA'},
    "Sugar Shack":       {'lat': 48.11914,  'lon': -96.18115,   'state': 'MN'},
    "Secret Factory":    {'lat': 41.446333, 'lon': -90.565487,  'state': 'IL'},
    "The Other Factory": {'lat': 35.1175,   'lon': -89.971107,  'state': 'TN'},
}
PRODUCT_FACTORY = {
    'Wonka Bar - Nutty Crunch Surprise':  "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows':          "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious':     "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate':         "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel':  "Wicked Choccy's",
    'Laffy Taffy':                        'Sugar Shack',
    'SweeTARTS':                          'Sugar Shack',
    'Nerds':                              'Sugar Shack',
    'Fun Dip':                            'Sugar Shack',
    'Fizzy Lifting Drinks':               'Sugar Shack',
    'Everlasting Gobstopper':             'Secret Factory',
    'Hair Toffee':                        'The Other Factory',
    'Lickable Wallpaper':                 'Secret Factory',
    'Wonka Gum':                          'Secret Factory',
    'Kazookles':                          'The Other Factory',
}
REGION_COORDS = {
    'Atlantic': {'lat': 38.9,  'lon': -77.0},
    'Interior': {'lat': 41.8,  'lon': -87.6},
    'Gulf':     {'lat': 29.7,  'lon': -95.4},
    'Pacific':  {'lat': 34.0,  'lon': -118.2},
}
PLOTLY_THEME = {
    'paper_bgcolor': '#0F1C14',
    'plot_bgcolor':  '#162519',
    'font_color':    '#95D5B2',
    'gridcolor':     'rgba(45,106,79,0.2)',
}
GREEN_SEQ = ['#1B4332','#2D6A4F','#40916C','#52B788','#74C69D','#95D5B2','#B7E4C7','#D8F3DC']

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return 2*R*asin(sqrt(a))

def apply_theme(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(family='Playfair Display', size=16, color='#fff')),
        paper_bgcolor=PLOTLY_THEME['paper_bgcolor'],
        plot_bgcolor=PLOTLY_THEME['plot_bgcolor'],
        font=dict(color=PLOTLY_THEME['font_color'], family='DM Sans'),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor='rgba(45,106,79,0.2)', zerolinecolor='rgba(45,106,79,0.2)')
    fig.update_yaxes(gridcolor='rgba(45,106,79,0.2)', zerolinecolor='rgba(45,106,79,0.2)')
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & ML (CACHED)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df = pd.read_csv('Nassau_Candy_Distributor.csv')
    feat = df.copy()
    feat['Order Date'] = pd.to_datetime(feat['Order Date'], dayfirst=True)
    feat['Ship Date']  = pd.to_datetime(feat['Ship Date'],  dayfirst=True)
    feat['Lead Time']        = (feat['Ship Date'] - feat['Order Date']).dt.days
    feat['Profit Margin %']  = feat['Gross Profit'] / feat['Sales'] * 100
    feat['Revenue per Unit'] = feat['Sales'] / feat['Units']
    feat['Cost per Unit']    = feat['Cost']  / feat['Units']
    feat['Factory']          = feat['Product Name'].map(PRODUCT_FACTORY)
    feat['Order Month']      = feat['Order Date'].dt.month
    feat['Order DayOfWeek']  = feat['Order Date'].dt.dayofweek
    feat['Factory Distance km'] = feat.apply(lambda r: haversine_km(
        FACTORIES[r['Factory']]['lat'], FACTORIES[r['Factory']]['lon'],
        REGION_COORDS[r['Region']]['lat'], REGION_COORDS[r['Region']]['lon']
    ) if r['Factory'] in FACTORIES and r['Region'] in REGION_COORDS else np.nan, axis=1)
    return df, feat

@st.cache_resource
def train_models(feat):
    ml_df = feat[['Product Name','Factory','Region','Ship Mode',
                  'Factory Distance km','Order Month','Order DayOfWeek',
                  'Revenue per Unit','Cost per Unit','Profit Margin %',
                  'Units','Lead Time']].dropna().copy()
    le_dict = {}
    for col in ['Product Name','Factory','Region','Ship Mode']:
        le = LabelEncoder()
        ml_df[col+'_enc'] = le.fit_transform(ml_df[col])
        le_dict[col] = le
    fcols = ['Product Name_enc','Factory_enc','Region_enc','Ship Mode_enc',
             'Factory Distance km','Order Month','Order DayOfWeek',
             'Revenue per Unit','Cost per Unit','Profit Margin %','Units']
    X = ml_df[fcols]; y = ml_df['Lead Time']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(200, max_depth=12, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
    }
    results = {}
    for name, m in models.items():
        if name == 'Linear Regression':
            m.fit(X_tr_s, y_tr); preds = m.predict(X_te_s)
        else:
            m.fit(X_tr, y_tr); preds = m.predict(X_te)
        results[name] = {
            'model': m, 'preds': preds,
            'RMSE': np.sqrt(mean_squared_error(y_te, preds)),
            'MAE':  mean_absolute_error(y_te, preds),
            'R2':   r2_score(y_te, preds),
            'y_test': y_te,
        }
    return results, le_dict, fcols, sc

@st.cache_data
def run_simulation(_feat, _le_dict, _fcols, _results):
    sim_model = _results['Gradient Boosting']['model']
    product_profile = _feat.groupby('Product Name').agg(
        avg_rev=('Revenue per Unit','median'), avg_cost=('Cost per Unit','median'),
        avg_margin=('Profit Margin %','median'), avg_units=('Units','median'),
        curr_factory=('Factory','first'),
    ).reset_index()
    recs = []
    for _, pr in product_profile.iterrows():
        for region in REGION_COORDS:
            for factory in FACTORIES:
                dist = haversine_km(FACTORIES[factory]['lat'], FACTORIES[factory]['lon'],
                                    REGION_COORDS[region]['lat'], REGION_COORDS[region]['lon'])
                for sm in _feat['Ship Mode'].unique():
                    recs.append({
                        'Product Name_enc':    _le_dict['Product Name'].transform([pr['Product Name']])[0],
                        'Factory_enc':         _le_dict['Factory'].transform([factory])[0],
                        'Region_enc':          _le_dict['Region'].transform([region])[0],
                        'Ship Mode_enc':       _le_dict['Ship Mode'].transform([sm])[0],
                        'Factory Distance km': dist,
                        'Order Month': 6, 'Order DayOfWeek': 2,
                        'Revenue per Unit': pr['avg_rev'], 'Cost per Unit': pr['avg_cost'],
                        'Profit Margin %': pr['avg_margin'], 'Units': pr['avg_units'],
                        'Product Name': pr['Product Name'], 'Factory': factory,
                        'Current Factory': pr['curr_factory'], 'Region': region,
                        'Ship Mode': sm, 'Distance km': dist, 'Profit Margin': pr['avg_margin'],
                    })
    sim_df = pd.DataFrame(recs)
    sim_df['Predicted Lead Time'] = sim_model.predict(sim_df[_fcols])
    current_lt = (_feat.groupby(['Product Name','Region','Ship Mode'])['Lead Time']
                  .mean().reset_index().rename(columns={'Lead Time':'Current Lead Time'}))
    best = (sim_df.sort_values('Predicted Lead Time')
            .groupby(['Product Name','Region','Ship Mode']).first().reset_index()
            [['Product Name','Region','Ship Mode','Factory','Distance km',
              'Predicted Lead Time','Profit Margin','Current Factory']])
    rec_df = best.merge(current_lt, on=['Product Name','Region','Ship Mode'], how='left')
    rec_df['Lead Time Reduction (Days)'] = (rec_df['Current Lead Time'] - rec_df['Predicted Lead Time']).round(2)
    rec_df['Lead Time Reduction (%)']    = (rec_df['Lead Time Reduction (Days)'] / rec_df['Current Lead Time'] * 100).round(2)
    rec_df['Is Reassignment'] = rec_df['Factory'] != rec_df['Current Factory']
    improvements = rec_df[rec_df['Is Reassignment'] & (rec_df['Lead Time Reduction (Days)'] > 0)].copy()
    product_rec = (improvements.groupby(['Product Name','Current Factory','Factory']).agg(
        Avg_LT_pct   = ('Lead Time Reduction (%)','mean'),
        Avg_LT_days  = ('Lead Time Reduction (Days)','mean'),
        Avg_Margin   = ('Profit Margin','mean'),
        Regions      = ('Region','nunique'),
    ).reset_index().sort_values('Avg_LT_pct', ascending=False).round(2))
    product_rec['Confidence'] = (
        0.5*(product_rec['Avg_LT_pct']/product_rec['Avg_LT_pct'].max()) +
        0.3*(product_rec['Avg_Margin']/product_rec['Avg_Margin'].max()) +
        0.2*(product_rec['Regions']/product_rec['Regions'].max())
    ).round(3)
    return improvements, product_rec, sim_df

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("🍫 Loading data & training models..."):
    df, feat = load_and_process()
    results, le_dict, fcols, scaler = train_models(feat)
    improvements, product_rec, sim_df = run_simulation(feat, le_dict, fcols, results)

best_model_name = min(results, key=lambda k: results[k]['RMSE'])

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🍫</div>
        <div style='font-family:Playfair Display,serif; font-size:1.1rem; font-weight:700; color:#fff;'>Nassau Candy</div>
        <div style='font-size:0.7rem; color:#52B788; letter-spacing:0.1em; text-transform:uppercase;'>Factory Optimizer</div>
    </div>
    <hr style='border-color:#2D6A4F44; margin:0.5rem 0 1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**🎛️ Filters**")
    sel_region    = st.selectbox("Customer Region", ["All"] + list(feat['Region'].unique()))
    sel_ship_mode = st.selectbox("Ship Mode",       ["All"] + list(feat['Ship Mode'].unique()))
    sel_division  = st.multiselect("Division", feat['Division'].unique(), default=list(feat['Division'].unique()))
    sel_product   = st.selectbox("Product (Simulator)", list(PRODUCT_FACTORY.keys()))

    st.markdown("<hr style='border-color:#2D6A4F44;'>", unsafe_allow_html=True)
    priority = st.slider("⚡ Optimization Priority", 0, 100, 50,
                         help="0 = maximize speed, 100 = maximize profit")
    st.caption(f"{'🚀 Speed Focus' if priority < 40 else '⚖️ Balanced' if priority < 70 else '💰 Profit Focus'}")

    st.markdown("<hr style='border-color:#2D6A4F44;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.75rem; color:#52B788;'>
    📊 <b>{len(df):,}</b> orders analyzed<br>
    🏭 <b>{len(FACTORIES)}</b> factories<br>
    🍬 <b>{len(PRODUCT_FACTORY)}</b> products<br>
    🗺️ <b>{feat['Region'].nunique()}</b> regions
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# APPLY SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
filtered = feat.copy()
if sel_region    != "All": filtered = filtered[filtered['Region'] == sel_region]
if sel_ship_mode != "All": filtered = filtered[filtered['Ship Mode'] == sel_ship_mode]
if sel_division:           filtered = filtered[filtered['Division'].isin(sel_division)]

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-title'>🍫 Factory Reallocation & Shipping Optimizer</div>
    <div class='hero-sub'>Nassau Candy Distributor · Intelligent Decision Intelligence System · Powered by ML</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
avg_lt       = filtered['Lead Time'].mean()
avg_margin   = filtered['Profit Margin %'].mean()
total_sales  = filtered['Sales'].sum()
total_profit = filtered['Gross Profit'].sum()
avg_lt_red   = improvements['Lead Time Reduction (%)'].mean()
best_r2      = results[best_model_name]['R2']
n_recs       = product_rec['Product Name'].nunique()
safe_recs    = (product_rec['Avg_Margin'] >= 60).sum()

k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
kpi_items = [
    (k1, f"{avg_lt:.1f}d",        "Avg Lead Time",         ""),
    (k2, f"{avg_margin:.1f}%",    "Avg Profit Margin",     ""),
    (k3, f"${total_sales/1e6:.1f}M", "Total Sales",        ""),
    (k4, f"${total_profit/1e6:.1f}M","Total Profit",       ""),
    (k5, f"{avg_lt_red:.1f}%",    "Potential LT Reduction","Sim result"),
    (k6, f"{best_r2:.3f}",        "Best Model R²",         best_model_name[:12]),
    (k7, f"{n_recs}",             "Products w/ Better Option",""),
    (k8, f"{safe_recs}",          "Safe Reassignments",    "Margin ≥ 60%"),
]
for col, val, label, delta in kpi_items:
    with col:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-value'>{val}</div>
            <div class='kpi-label'>{label}</div>
            {'<div class="kpi-delta">' + delta + '</div>' if delta else ''}
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA Overview",
    "🤖 Model Performance",
    "🏭 Factory Simulator",
    "🏆 Recommendations",
    "⚠️ Risk & Impact",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EDA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        # Sales by Division
        div_s = filtered.groupby('Division')['Sales'].sum().reset_index()
        fig = px.bar(div_s, x='Division', y='Sales', color='Sales',
                     color_continuous_scale=GREEN_SEQ, text_auto='.2s')
        fig.update_traces(textposition='outside')
        apply_theme(fig, "💰 Total Sales by Division")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        # Lead Time by Ship Mode
        lt_ship = filtered.groupby('Ship Mode')['Lead Time'].mean().reset_index().sort_values('Lead Time', ascending=True)
        fig2 = px.bar(lt_ship, x='Lead Time', y='Ship Mode', orientation='h',
                      color='Lead Time', color_continuous_scale=GREEN_SEQ)
        apply_theme(fig2, "⏱️ Avg Lead Time by Ship Mode")
        fig2.update_coloraxes(showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        # Profit Margin by Product
        pm = filtered.groupby('Product Name')['Profit Margin %'].mean().reset_index().sort_values('Profit Margin %', ascending=True)
        fig3 = px.bar(pm, x='Profit Margin %', y='Product Name', orientation='h',
                      color='Profit Margin %', color_continuous_scale=GREEN_SEQ)
        apply_theme(fig3, "📈 Avg Profit Margin by Product", height=500)
        fig3.update_coloraxes(showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

        # Ship Mode pie
        sc_data = filtered['Ship Mode'].value_counts().reset_index()
        sc_data.columns = ['Ship Mode', 'Count']
        fig4 = px.pie(sc_data, values='Count', names='Ship Mode',
                      color_discrete_sequence=GREEN_SEQ, hole=0.45)
        apply_theme(fig4, "🚚 Order Share by Ship Mode", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # Lead Time Heatmap — Product × Region
    st.markdown("<div class='section-header'>Lead Time Heatmap — Product × Region</div>", unsafe_allow_html=True)
    lt_heat = filtered.pivot_table(values='Lead Time', index='Product Name', columns='Region', aggfunc='mean').round(1)
    fig5 = go.Figure(go.Heatmap(
        z=lt_heat.values, x=lt_heat.columns.tolist(), y=lt_heat.index.tolist(),
        colorscale=[[0,'#1B4332'],[0.5,'#52B788'],[1,'#C0392B']],
        text=lt_heat.values, texttemplate='%{text}', textfont={"size":11},
        hoverongaps=False,
    ))
    apply_theme(fig5, "🌡️ Average Lead Time (Days) — Darker Red = Slower Route", height=480)
    st.plotly_chart(fig5, use_container_width=True)

    # Monthly trend
    st.markdown("<div class='section-header'>Monthly Sales & Profit Trend</div>", unsafe_allow_html=True)
    trend = filtered.copy()
    trend['YearMonth'] = trend['Order Date'].dt.to_period('M').astype(str)
    monthly = trend.groupby('YearMonth').agg(Sales=('Sales','sum'), Profit=('Gross Profit','sum')).reset_index()
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Sales'],
                              name='Sales', line=dict(color='#52B788', width=2), fill='tozeroy',
                              fillcolor='rgba(82,183,136,0.15)'), secondary_y=False)
    fig6.add_trace(go.Scatter(x=monthly['YearMonth'], y=monthly['Profit'],
                              name='Gross Profit', line=dict(color='#C9A84C', width=2, dash='dot')), secondary_y=True)
    fig6.update_xaxes(tickangle=45, nticks=20)
    apply_theme(fig6, "📅 Monthly Sales & Profit", height=360)
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Predictive Model Comparison</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    metrics_data = []
    for name in results:
        metrics_data.append({'Model': name,
                             'RMSE': results[name]['RMSE'],
                             'MAE':  results[name]['MAE'],
                             'R²':   results[name]['R2']})
    mdf = pd.DataFrame(metrics_data)

    for col, metric, better in zip([c1, c2, c3], ['RMSE','MAE','R²'], ['low','low','high']):
        with col:
            fig = px.bar(mdf, x='Model', y=metric,
                         color=metric, color_continuous_scale=GREEN_SEQ if better == 'high' else GREEN_SEQ[::-1])
            fig.update_traces(text=mdf[metric].round(3), textposition='outside')
            apply_theme(fig, f"{'📉' if better=='low' else '📈'} {metric} ({'lower' if better=='low' else 'higher'} = better)")
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Actual vs Predicted scatter for best model
    st.markdown("<div class='section-header'>Actual vs Predicted Lead Time</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        y_test  = results[best_model_name]['y_test']
        y_pred  = results[best_model_name]['preds']
        scatter_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        fig7 = px.scatter(scatter_df, x='Actual', y='Predicted',
                          opacity=0.4, color_discrete_sequence=['#52B788'])
        min_v, max_v = scatter_df['Actual'].min(), scatter_df['Actual'].max()
        fig7.add_trace(go.Scatter(x=[min_v,max_v], y=[min_v,max_v],
                                  mode='lines', line=dict(color='#C9A84C', dash='dash'),
                                  name='Perfect Prediction'))
        apply_theme(fig7, f"🎯 Actual vs Predicted — {best_model_name}", height=420)
        st.plotly_chart(fig7, use_container_width=True)

    with c2:
        # Feature importance for tree model
        tree_name  = 'Gradient Boosting' if results['Gradient Boosting']['R2'] >= results['Random Forest']['R2'] else 'Random Forest'
        tree_model = results[tree_name]['model']
        fi_df = pd.DataFrame({'Feature': fcols, 'Importance': tree_model.feature_importances_}).sort_values('Importance')
        fig8 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=GREEN_SEQ)
        apply_theme(fig8, f"🔍 Feature Importance — {tree_name}", height=420)
        fig8.update_coloraxes(showscale=False)
        st.plotly_chart(fig8, use_container_width=True)

    # Model metrics table
    st.markdown("<div class='section-header'>Detailed Metrics</div>", unsafe_allow_html=True)
    styled_mdf = mdf.copy()
    styled_mdf['Best'] = [('✅' if n == best_model_name else '') for n in styled_mdf['Model']]
    st.dataframe(styled_mdf.style.background_gradient(subset=['RMSE','MAE'], cmap='RdYlGn_r')
                                  .background_gradient(subset=['R²'], cmap='RdYlGn')
                                  .format({'RMSE':'{:.3f}','MAE':'{:.3f}','R²':'{:.4f}'}),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — FACTORY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>What-If Factory Simulator</div>", unsafe_allow_html=True)
    st.caption("Select a product and see predicted lead time performance across all factory options.")

    prod_sim = sim_df[sim_df['Product Name'] == sel_product].copy()
    curr_factory = PRODUCT_FACTORY[sel_product]

    c1, c2 = st.columns([2, 1])
    with c1:
        # Avg predicted lead time per factory × region
        sim_agg = prod_sim.groupby(['Factory','Region'])['Predicted Lead Time'].mean().reset_index()
        sim_agg['Is Current'] = sim_agg['Factory'] == curr_factory

        fig9 = px.bar(sim_agg, x='Factory', y='Predicted Lead Time', color='Region',
                      barmode='group', color_discrete_sequence=GREEN_SEQ,
                      pattern_shape='Is Current',
                      pattern_shape_map={True: '/', False: ''})
        fig9.add_hline(y=sim_agg[sim_agg['Factory']==curr_factory]['Predicted Lead Time'].mean(),
                       line_dash='dash', line_color='#C9A84C',
                       annotation_text=f'Current ({curr_factory})',
                       annotation_position='top left')
        apply_theme(fig9, f"📦 {sel_product} — Predicted Lead Time by Factory & Region", height=420)
        st.plotly_chart(fig9, use_container_width=True)

    with c2:
        # Best option callout
        best_opt = sim_agg.sort_values('Predicted Lead Time').iloc[0]
        curr_avg = sim_agg[sim_agg['Factory']==curr_factory]['Predicted Lead Time'].mean()
        best_avg = sim_agg.groupby('Factory')['Predicted Lead Time'].mean().min()
        reduction = ((curr_avg - best_avg) / curr_avg * 100)

        best_factory_name = sim_agg.groupby('Factory')['Predicted Lead Time'].mean().idxmin()

        st.markdown(f"""
        <div class='kpi-card' style='margin-top:0.5rem; text-align:left; padding:1.4rem;'>
            <div style='font-size:0.7rem; color:#52B788; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>Current Assignment</div>
            <div style='font-family:Playfair Display,serif; font-size:1.2rem; color:#fff; font-weight:700;'>{curr_factory}</div>
            <div style='color:#78A98A; font-size:0.85rem; margin-top:0.3rem;'>Avg Lead Time: <b>{curr_avg:.1f} days</b></div>
        </div>
        <div class='kpi-card' style='margin-top:0.7rem; text-align:left; padding:1.4rem; border-color:#40916C;'>
            <div style='font-size:0.7rem; color:#C9A84C; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.5rem;'>✨ Recommended Factory</div>
            <div style='font-family:Playfair Display,serif; font-size:1.2rem; color:#fff; font-weight:700;'>{best_factory_name}</div>
            <div style='color:#78A98A; font-size:0.85rem; margin-top:0.3rem;'>Avg Lead Time: <b>{best_avg:.1f} days</b></div>
            <div style='color:#C9A84C; font-size:0.9rem; margin-top:0.5rem; font-weight:600;'>📉 {reduction:.1f}% faster</div>
        </div>
        """, unsafe_allow_html=True)

    # Ship mode breakdown
    st.markdown("<div class='section-header'>Lead Time by Ship Mode × Factory</div>", unsafe_allow_html=True)
    sm_agg = prod_sim.groupby(['Factory','Ship Mode'])['Predicted Lead Time'].mean().reset_index()
    fig10 = px.line(sm_agg, x='Ship Mode', y='Predicted Lead Time', color='Factory',
                    markers=True, color_discrete_sequence=GREEN_SEQ)
    apply_theme(fig10, f"🚚 {sel_product} — Ship Mode Impact on Lead Time", height=380)
    st.plotly_chart(fig10, use_container_width=True)

    # Factory map
    st.markdown("<div class='section-header'>Factory Locations Map</div>", unsafe_allow_html=True)
    factory_map_df = pd.DataFrame([
        {'Factory': k, 'lat': v['lat'], 'lon': v['lon'],
         'Is Current': k == curr_factory,
         'Label': f"{'⭐ ' if k==curr_factory else ''}{k}"}
        for k, v in FACTORIES.items()
    ])
    factory_map_df['Color'] = factory_map_df['Is Current'].map({True:'#C9A84C', False:'#52B788'})
    factory_map_df['Size']  = factory_map_df['Is Current'].map({True:20, False:10})
    fig_map = px.scatter_mapbox(factory_map_df, lat='lat', lon='lon', text='Label',
                                color='Is Current', size='Size',
                                color_discrete_map={True:'#C9A84C', False:'#52B788'},
                                mapbox_style='carto-darkmatter', zoom=3, height=400)
    fig_map.update_layout(paper_bgcolor='#0F1C14', margin=dict(l=0,r=0,t=0,b=0),
                          showlegend=False)
    st.plotly_chart(fig_map, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>🏆 Top Factory Reassignment Recommendations</div>", unsafe_allow_html=True)

    # Priority-weighted sort
    speed_w  = 1 - priority / 100
    profit_w = priority / 100
    ranked = product_rec.copy()
    ranked['Priority Score'] = (
        speed_w  * (ranked['Avg_LT_pct']  / ranked['Avg_LT_pct'].max()) +
        profit_w * (ranked['Avg_Margin']  / ranked['Avg_Margin'].max())
    ).round(3)
    ranked = ranked.sort_values('Priority Score', ascending=False)

    # Recommendation cards
    for i, (_, row) in enumerate(ranked.head(10).iterrows()):
        risk = '✅ Safe' if row['Avg_Margin'] >= 60 else '⚠️ Risk'
        badge_cls = 'badge-safe' if row['Avg_Margin'] >= 60 else 'badge-risk'
        st.markdown(f"""
        <div class='rec-card'>
            <div class='rec-rank'>#{i+1}</div>
            <div class='rec-body'>
                <div class='rec-product'>🍬 {row['Product Name']}</div>
                <div class='rec-detail'>
                    {row['Current Factory']} → <b style='color:#95D5B2;'>{row['Factory']}</b>
                    &nbsp;·&nbsp; {row['Avg_LT_pct']:.1f}% faster &nbsp;·&nbsp; {row['Avg_LT_days']:.0f} days saved
                    &nbsp;·&nbsp; Margin: {row['Avg_Margin']:.1f}% &nbsp;·&nbsp; {int(row['Regions'])} region(s)
                </div>
            </div>
            <div class='rec-badge' style='margin-right:0.5rem;'>Conf: {row['Confidence']:.2f}</div>
            <div class='rec-badge {badge_cls}'>{risk}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig11 = px.bar(ranked.head(10), x='Avg_LT_pct', y='Product Name',
                       orientation='h', color='Confidence',
                       color_continuous_scale=GREEN_SEQ,
                       text=ranked.head(10)['Avg_LT_pct'].round(1).astype(str) + '%')
        fig11.update_traces(textposition='outside')
        apply_theme(fig11, "📉 Lead Time Reduction % by Product", height=420)
        st.plotly_chart(fig11, use_container_width=True)

    with c2:
        fig12 = px.scatter(product_rec, x='Avg_LT_pct', y='Confidence',
                           size='Avg_Margin', color='Avg_Margin',
                           hover_name='Product Name',
                           hover_data={'Factory': True, 'Regions': True},
                           color_continuous_scale=GREEN_SEQ,
                           size_max=30)
        apply_theme(fig12, "🎯 Confidence vs Lead Time Reduction (bubble = margin)", height=420)
        st.plotly_chart(fig12, use_container_width=True)

    # Full table
    st.markdown("<div class='section-header'>Full Recommendation Table</div>", unsafe_allow_html=True)
    display_cols = {'Product Name':'Product','Current Factory':'From','Factory':'To →',
                    'Avg_LT_pct':'LT Reduction (%)','Avg_LT_days':'Days Saved',
                    'Avg_Margin':'Margin (%)','Regions':'Regions','Confidence':'Confidence'}
    show_df = ranked.rename(columns=display_cols)[list(display_cols.values())]
    st.dataframe(show_df.style
                 .background_gradient(subset=['LT Reduction (%)'], cmap='Greens')
                 .background_gradient(subset=['Margin (%)'], cmap='YlOrRd')
                 .format({'LT Reduction (%)':'{:.1f}','Days Saved':'{:.0f}',
                          'Margin (%)':'{:.1f}','Confidence':'{:.3f}'}),
                 use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — RISK & IMPACT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>Risk & Profit Impact Analysis</div>", unsafe_allow_html=True)

    profit_impact = improvements.groupby(['Product Name','Factory']).agg(
        Avg_Margin     = ('Profit Margin','mean'),
        LT_Gain_pct    = ('Lead Time Reduction (%)','mean'),
        Total_Orders   = ('Region','count'),
    ).reset_index().round(2)
    profit_impact['Risk'] = profit_impact['Avg_Margin'].apply(
        lambda m: '⚠️ Low Margin Risk' if m < 60 else '✅ Safe'
    )

    c1, c2 = st.columns(2)
    with c1:
        colors = profit_impact['Risk'].map({'✅ Safe':'#52B788','⚠️ Low Margin Risk':'#C0392B'})
        fig13 = go.Figure()
        for risk_type, color in [('✅ Safe','#52B788'),('⚠️ Low Margin Risk','#C0392B')]:
            sub = profit_impact[profit_impact['Risk']==risk_type]
            fig13.add_trace(go.Scatter(
                x=sub['LT_Gain_pct'], y=sub['Avg_Margin'],
                mode='markers+text', name=risk_type,
                text=sub['Product Name'].str[:12],
                textposition='top center', textfont=dict(size=8),
                marker=dict(size=12, color=color, opacity=0.85,
                            line=dict(width=1, color='#fff')),
            ))
        fig13.add_hline(y=60, line_dash='dash', line_color='#C9A84C',
                        annotation_text='Margin Threshold (60%)')
        fig13.add_vline(x=profit_impact['LT_Gain_pct'].median(), line_dash='dash',
                        line_color='#52B788', annotation_text='Median Gain')
        apply_theme(fig13, "⚖️ Risk Quadrant: Speed Gain vs Profit Safety", height=480)
        st.plotly_chart(fig13, use_container_width=True)

    with c2:
        # Route clustering
        route_df = feat.groupby(['Factory','Region']).agg(
            Avg_Lead_Time  = ('Lead Time','mean'),
            Avg_Distance   = ('Factory Distance km','mean'),
            Avg_Margin     = ('Profit Margin %','mean'),
            Total_Orders   = ('Row ID','count'),
        ).reset_index().round(2)
        km = KMeans(n_clusters=3, random_state=42, n_init='auto')
        route_df['Cluster'] = km.fit_predict(route_df[['Avg_Lead_Time','Avg_Distance','Avg_Margin']])
        cmeans = route_df.groupby('Cluster')['Avg_Lead_Time'].mean()
        cmap   = {cmeans.idxmin():'Fast 🟢', cmeans.idxmax():'Slow 🔴'}
        rem    = [c for c in cmeans.index if c not in cmap]
        if rem: cmap[rem[0]] = 'Medium 🟡'
        route_df['Route Type'] = route_df['Cluster'].map(cmap)

        fig14 = px.scatter(route_df, x='Avg_Distance', y='Avg_Lead_Time',
                           size='Total_Orders', color='Route Type',
                           hover_name='Factory',
                           hover_data={'Region':True,'Avg_Margin':True},
                           color_discrete_map={'Fast 🟢':'#2D6A4F','Medium 🟡':'#D4A017','Slow 🔴':'#C0392B'},
                           size_max=40, text='Region')
        fig14.update_traces(textposition='top center', textfont=dict(size=8))
        apply_theme(fig14, "🗺️ Route Cluster Analysis (bubble = order volume)", height=480)
        st.plotly_chart(fig14, use_container_width=True)

    # Risk summary metrics
    st.markdown("<div class='section-header'>Risk Summary</div>", unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    safe_n   = (profit_impact['Risk']=='✅ Safe').sum()
    risky_n  = (profit_impact['Risk']=='⚠️ Low Margin Risk').sum()
    avg_gain = profit_impact['LT_Gain_pct'].mean()
    top_gain = profit_impact['LT_Gain_pct'].max()

    for col, val, label, color in [
        (r1, f"{safe_n}", "Safe Reassignments ✅",  "#2D6A4F"),
        (r2, f"{risky_n}","Risky Reassignments ⚠️","#7b1a13"),
        (r3, f"{avg_gain:.1f}%", "Avg Lead Time Gain","#1a3a28"),
        (r4, f"{top_gain:.1f}%", "Max Lead Time Gain","#1a3a28"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='border-color:{color}88;'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # Margin distribution by recommended factory
    st.markdown("<div class='section-header'>Profit Margin Distribution by Recommended Factory</div>", unsafe_allow_html=True)
    fig15 = px.box(improvements, x='Factory', y='Profit Margin',
                   color='Factory', color_discrete_sequence=GREEN_SEQ,
                   points='outliers')
    apply_theme(fig15, "📊 Profit Margin Stability Across Recommended Factories", height=400)
    fig15.update_layout(showlegend=False)
    st.plotly_chart(fig15, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2D6A4F44; margin-top:2rem;'>
<div style='text-align:center; color:#52B788; font-size:0.75rem; padding:0.5rem 0 1rem 0;'>
    🍫 Nassau Candy Distributor · Factory Reallocation & Shipping Optimization System · Built with Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
