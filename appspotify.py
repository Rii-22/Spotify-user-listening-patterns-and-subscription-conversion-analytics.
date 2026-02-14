import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import re

# ==========================================
# 0. STREAMLIT UI CONFIGURATION (Spotify Theme)
# ==========================================
st.set_page_config(page_title="Spotify Growth Engine", layout="wide")

# Custom CSS for Spotify Aesthetics
st.markdown("""
    <style>
    .main { background-color: #191414; color: #FFFFFF; }
    .stMetric { background-color: #282828; padding: 15px; border-radius: 10px; border-left: 5px solid #1DB954; }
    div[data-testid="stMetricValue"] { color: #1DB954; }
    .stButton>button { background-color: #1DB954; color: white; border-radius: 20px; width: 100%; }
    .stHeader { color: #1DB954; }
    h1, h2, h3 { color: #1DB954 !important; }
    .stDataFrame { background-color: #282828; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. ADVANCED DATA SIMULATION ENGINE
# ==========================================
@st.cache_data
def load_behavioral_data(n=10000):
    np.random.seed(42)
    genres = ['Pop', 'Indie', 'Techno', 'Lo-Fi', 'Jazz', 'Rock']
    devices = ['iOS_iPhone15_Mobile', 'Android_S23_Mobile', 'Win11_Desktop_App', 'MacOS_Sonoma_Desktop']
    
    data = {
        'User_ID': [f"USR-{i:05d}" for i in range(n)],
        'Age': np.random.normal(28, 8, n).astype(int).clip(13, 65),
        'Tier': np.random.choice(['Free', 'Premium'], n, p=[0.65, 0.35]),
        'Device_Log': np.random.choice(devices, n),
        'Daily_Avg_Mins': np.random.gamma(shape=2, scale=20, size=n), 
        'Playlists_Created': np.random.poisson(lam=3, size=n),
        'Top_Genre': np.random.choice(genres, n),
    }
    
    df = pd.DataFrame(data)
    
    # Behavioral Logic: Free users have higher skip rates (pain point)
    df['Skip_Rate'] = df['Tier'].apply(lambda x: np.random.uniform(10, 50) if x == 'Free' else np.random.uniform(0, 5))
    
    # Conversion Logic: High engagement + High frustration = High Conversion Probability
    prob = (df['Daily_Avg_Mins'] / df['Daily_Avg_Mins'].max()) * 0.5 + (df['Skip_Rate'] / 50) * 0.5
    df['Converted_Last_30D'] = np.where((df['Tier'] == 'Free') & (prob > 0.6), True, False)
    
    return df

df = load_behavioral_data()

# ==========================================
# 2. STRATEGIC ANALYTICAL MODULES
# ==========================================

# A. Regex Device Categorization (The Engineering POV)
df['Is_Mobile'] = df['Device_Log'].apply(lambda x: bool(re.search(r'Mobile', x, re.IGNORECASE)))

# B. The "Frustration Index" (The Product POV)
df['Frustration_Index'] = (df['Daily_Avg_Mins'] * df['Skip_Rate']) / 100

# C. Statistical Power (The BA POV)
premium_mins = df[df['Tier'] == 'Premium']['Daily_Avg_Mins']
free_mins = df[df['Tier'] == 'Free']['Daily_Avg_Mins']
t_stat, p_val = stats.ttest_ind(premium_mins, free_mins)

# ==========================================
# 3. INTERACTIVE DASHBOARD UI
# ==========================================
st.title("♫ Spotify Product Growth Dashboard")
st.subheader("Conversion Triggers & Behavioral Friction Analysis")

# Sidebar Controls
st.sidebar.header("Campaign Targeting")
risk_threshold = st.sidebar.slider("Frustration Index Threshold", 0.0, 50.0, 25.0)
target_genre = st.sidebar.multiselect("Filter Genres", options=df['Top_Genre'].unique(), default=df['Top_Genre'].unique())

filtered_df = df[df['Top_Genre'].isin(target_genre)]

# --- ROW 1: KPI CARDS ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Users", f"{len(df):,}")
with col2:
    conv_rate = (df['Converted_Last_30D'].sum() / len(df[df['Tier'] == 'Free'])) * 100
    st.metric("Conversion Rate", f"{conv_rate:.1f}%")
with col3:
    st.metric("Avg Mins/User", f"{df['Daily_Avg_Mins'].mean():.1f}")
with col4:
    st.metric("Stat. P-Value", f"{p_val:.2e}")

# --- ROW 2: VISUAL ANALYTICS ---
st.markdown("---")
left_chart, right_chart = st.columns(2)

with left_chart:
    st.write("### Conversion Rate by Music Genre")
    genre_conv = filtered_df.groupby('Top_Genre')['Converted_Last_30D'].mean().sort_values()
    st.bar_chart(genre_conv, color="#1DB954")

with right_chart:
    st.write("### The 'Conversion Zone' Analysis")
    # Scatter plot showing where Free users become Prime for conversion
    scatter_data = filtered_df.sample(min(1000, len(filtered_df)))
    st.scatter_chart(scatter_data, x='Daily_Avg_Mins', y='Skip_Rate', color='Tier')

# --- ROW 3: TARGETING ENGINE ---
st.markdown("---")
st.write("### 🎯 High-Value Conversion Targets")
st.caption(f"Users currently on FREE tier with a Frustration Index > {risk_threshold}")
high_value_targets = filtered_df[(filtered_df['Tier'] == 'Free') & (filtered_df['Frustration_Index'] > risk_threshold)]

st.dataframe(
    high_value_targets[['User_ID', 'Top_Genre', 'Daily_Avg_Mins', 'Skip_Rate', 'Frustration_Index']]
    .sort_values(by='Frustration_Index', ascending=False), 
    use_container_width=True
)

# ==========================================
# 4. EXECUTIVE SUMMARY
# ==========================================
st.success("### Senior Lead Insights")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown(f"""
    **Behavioral Significance:**
    The P-Value of `{p_val:.2e}` suggests that moving users to Premium significantly increases their daily listening time. 
    **Action:** Prioritize the conversion of 'Heavy Listeners' over 'High-Skip' listeners to maximize long-term Lifetime Value (LTV).
    """)

with col_b:
    mobile_conv = df[df['Is_Mobile'] == True]['Converted_Last_30D'].mean()
    st.markdown(f"""
    **Unique Insight (Mobile vs. Desktop):**
    Mobile users are currently converting at a rate of **{mobile_conv:.1%}**. 
    **Recommendation:** Implement an 'Ad-Free Mobile' trial specifically targeted at the **{genre_conv.index[-1]}** listeners to capitalize on high mobile engagement.
    """)
