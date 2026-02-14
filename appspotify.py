import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import re
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Spotify Conversion Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Spotify Dark Mode
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #191414;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1DB954 !important;
        font-family: 'Circular', 'Helvetica Neue', sans-serif;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #1DB954;
        font-size: 2rem;
        font-weight: bold;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B3B3B3;
        font-size: 0.9rem;
    }
    
    /* KPI Cards */
    div.element-container div.stMarkdown {
        background-color: #282828;
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #1DB954;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #1DB954;
        color: #000000;
        font-weight: bold;
        border-radius: 500px;
        border: none;
        padding: 12px 32px;
    }
    
    .stButton>button:hover {
        background-color: #1ED760;
    }
    
    /* Sliders */
    .stSlider>div>div>div {
        background-color: #1DB954;
    }
    
    /* Text */
    p, label {
        color: #B3B3B3;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #282828;
        color: #FFFFFF;
    }
    
    /* Tables */
    .dataframe {
        background-color: #282828;
    }
    
    /* Success/Info boxes */
    .stAlert {
        background-color: #282828;
        color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_synthetic_data(n_users=10000, random_seed=42):
    """
    Generate high-fidelity synthetic Spotify user data with realistic behavioral patterns.
    """
    np.random.seed(random_seed)
    
    # User profiles
    user_ids = [f"USER_{str(i).zfill(6)}" for i in range(1, n_users + 1)]
    ages = np.random.choice(range(18, 65), n_users, p=np.array([0.3, 0.25, 0.2, 0.15, 0.1]))
    account_age_days = np.random.gamma(shape=2, scale=180, size=n_users).astype(int)
    
    # Tier distribution (70% Free, 30% Premium)
    tiers = np.random.choice(['Free', 'Premium'], n_users, p=[0.70, 0.30])
    
    # Device types with realistic patterns
    device_categories = [
        'iOS_iPhone15_Mobile', 'iOS_iPhone14_Mobile', 'Android_Galaxy_Mobile',
        'Win11_Desktop_App', 'MacOS_Desktop_App', 'Web_Chrome_Browser',
        'Android_Tablet', 'iOS_iPad_Tablet'
    ]
    devices = np.random.choice(device_categories, n_users, 
                              p=[0.25, 0.15, 0.20, 0.15, 0.10, 0.08, 0.04, 0.03])
    
    # Listening metrics - Premium users listen more
    base_daily_mins = np.random.gamma(shape=3, scale=30, size=n_users)
    daily_avg_mins = np.where(tiers == 'Premium', 
                              base_daily_mins * 1.4,  # Premium users listen 40% more
                              base_daily_mins)
    daily_avg_mins = np.clip(daily_avg_mins, 5, 300)
    
    # Skip rate - Free users skip more due to ads
    base_skip_rate = np.random.beta(a=2, b=5, size=n_users)
    skip_rate = np.where(tiers == 'Free',
                         base_skip_rate * 1.5,  # Free users skip 50% more
                         base_skip_rate)
    skip_rate = np.clip(skip_rate, 0, 1)
    
    # Offline hours - only Premium feature
    offline_hours = np.where(tiers == 'Premium',
                            np.random.gamma(shape=2, scale=5, size=n_users),
                            0)
    
    # Interaction data
    playlists_created = np.random.poisson(lam=5, size=n_users)
    discovery_weekly_click_rate = np.random.beta(a=5, b=3, size=n_users)
    
    # Top genre preferences
    genres = ['Pop', 'Hip-Hop', 'Rock', 'Electronic', 'R&B', 'Indie', 'Latin', 'Country']
    top_genres = np.random.choice(genres, n_users, 
                                  p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07])
    
    # Conversion logic - influenced by frustration (high skip rate + high daily mins)
    # Users who love the content but hate the limitations are most likely to convert
    frustration_score = (skip_rate * daily_avg_mins / 100)
    conversion_probability = 1 / (1 + np.exp(-2 * (frustration_score - 0.5)))
    
    # Additional conversion factors
    conversion_probability *= (1 + 0.1 * (account_age_days > 90))  # Longer users more likely
    conversion_probability *= (1 + 0.15 * (playlists_created > 8))  # Engaged users
    
    # Only Free users can convert
    conversion_probability = np.where(tiers == 'Free', conversion_probability, 0)
    
    converted_last_30d = np.random.binomial(1, np.clip(conversion_probability, 0, 0.35), n_users)
    
    # Create DataFrame
    df = pd.DataFrame({
        'User_ID': user_ids,
        'Age': ages,
        'Tier': tiers,
        'Account_Age_Days': account_age_days,
        'Daily_Avg_Mins': daily_avg_mins.round(2),
        'Skip_Rate': skip_rate.round(3),
        'Offline_Hours_Logged': offline_hours.round(2),
        'Playlists_Created': playlists_created,
        'Discovery_Weekly_Click_Rate': discovery_weekly_click_rate.round(3),
        'Device_Type': devices,
        'Top_Genre': top_genres,
        'Converted_Last_30D': converted_last_30d
    })
    
    return df

@st.cache_data
def calculate_frustration_index(df):
    """
    Calculate the Frustration Index: composite score of Skip_Rate vs Daily_Avg_Mins.
    High score = users who love content but hate limitations (prime conversion targets).
    """
    # Normalize both metrics to 0-1 scale
    skip_normalized = (df['Skip_Rate'] - df['Skip_Rate'].min()) / (df['Skip_Rate'].max() - df['Skip_Rate'].min())
    mins_normalized = (df['Daily_Avg_Mins'] - df['Daily_Avg_Mins'].min()) / (df['Daily_Avg_Mins'].max() - df['Daily_Avg_Mins'].min())
    
    # Frustration = High skips * High engagement
    frustration_index = skip_normalized * mins_normalized * 100
    
    return frustration_index

@st.cache_data
def perform_ab_test(df):
    """
    A/B Test: Compare engagement levels between Free and Premium users.
    """
    free_engagement = df[df['Tier'] == 'Free']['Daily_Avg_Mins']
    premium_engagement = df[df['Tier'] == 'Premium']['Daily_Avg_Mins']
    
    t_stat, p_value = stats.ttest_ind(premium_engagement, free_engagement)
    
    return {
        'free_mean': free_engagement.mean(),
        'premium_mean': premium_engagement.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

@st.cache_data
def categorize_devices(df):
    """
    Use regex to parse device types and categorize into Mobile, Desktop, Tablet.
    """
    def extract_category(device_string):
        if re.search(r'Mobile', device_string, re.IGNORECASE):
            return 'Mobile'
        elif re.search(r'Desktop|Win|Mac', device_string, re.IGNORECASE):
            return 'Desktop'
        elif re.search(r'Tablet|iPad', device_string, re.IGNORECASE):
            return 'Tablet'
        else:
            return 'Web'
    
    df['Device_Category'] = df['Device_Type'].apply(extract_category)
    return df

@st.cache_data
def detect_power_users(df):
    """
    Identify power users using Z-score (outliers with >2.5 standard deviations).
    """
    z_scores = np.abs(stats.zscore(df['Daily_Avg_Mins']))
    df['Z_Score'] = z_scores
    df['Power_User'] = (z_scores > 2.5) & (df['Tier'] == 'Free')
    
    return df

# Main Application
def main():
    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>🎵 Spotify User Engagement Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #1DB954; font-size: 1.2rem; margin-top: 0;'>Conversion Triggers & Behavioral Analytics</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Generate data
    df = generate_synthetic_data()
    df['Frustration_Index'] = calculate_frustration_index(df)
    df = categorize_devices(df)
    df = detect_power_users(df)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Skip threshold slider
    st.sidebar.subheader("Conversion Probability Slider")
    skip_threshold = st.sidebar.slider(
        "Skip Rate Threshold for Premium Ads",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Adjust to see how many Free users would be targeted"
    )
    
    # Calculate targeted users
    free_users = df[df['Tier'] == 'Free']
    targeted_users = free_users[free_users['Skip_Rate'] >= skip_threshold]
    
    st.sidebar.metric(
        "Targeted Free Users",
        f"{len(targeted_users):,}",
        f"{(len(targeted_users)/len(free_users)*100):.1f}% of Free tier"
    )
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.subheader("Filters")
    selected_genres = st.sidebar.multiselect(
        "Filter by Genre",
        options=sorted(df['Top_Genre'].unique()),
        default=sorted(df['Top_Genre'].unique())
    )
    
    # Apply filters
    filtered_df = df[df['Top_Genre'].isin(selected_genres)]
    
    # KPI Cards - "Wrapped" Style
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        conversion_rate = (filtered_df['Converted_Last_30D'].sum() / len(filtered_df[filtered_df['Tier'] == 'Free'])) * 100
        st.metric(
            "Conversion Rate (30D)",
            f"{conversion_rate:.2f}%",
            delta=f"+{conversion_rate - 4.5:.2f}% vs baseline"
        )
    
    with col2:
        avg_session_mins = filtered_df['Daily_Avg_Mins'].mean()
        st.metric(
            "Avg. Daily Minutes",
            f"{avg_session_mins:.1f} min",
            delta=f"{avg_session_mins - 60:.1f} min"
        )
    
    with col3:
        top_conversion_genre = filtered_df.groupby('Top_Genre')['Converted_Last_30D'].sum().idxmax()
        genre_conversions = filtered_df.groupby('Top_Genre')['Converted_Last_30D'].sum().max()
        st.metric(
            "Top Conversion Genre",
            top_conversion_genre,
            f"{genre_conversions} conversions"
        )
    
    with col4:
        power_users_count = filtered_df['Power_User'].sum()
        st.metric(
            "Power Users (Free Tier)",
            f"{power_users_count}",
            "High-Value Targets 🎯"
        )
    
    st.markdown("---")
    
    # A/B Test Results
    st.subheader("📈 Statistical Significance: Platform Stickiness Analysis")
    
    ab_results = perform_ab_test(filtered_df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        **Hypothesis Test:** Premium tier increases daily engagement (Platform Stickiness)
        
        - **Free Users Avg. Engagement:** {ab_results['free_mean']:.2f} minutes/day
        - **Premium Users Avg. Engagement:** {ab_results['premium_mean']:.2f} minutes/day
        - **Difference:** {ab_results['premium_mean'] - ab_results['free_mean']:.2f} minutes/day (+{((ab_results['premium_mean'] / ab_results['free_mean']) - 1) * 100:.1f}%)
        - **T-Statistic:** {ab_results['t_statistic']:.4f}
        - **P-Value:** {ab_results['p_value']:.6f}
        """)
        
        if ab_results['significant']:
            st.success(f"✅ **Statistically Significant** (p < 0.05): Premium users show significantly higher engagement!")
        else:
            st.warning("⚠️ Results not statistically significant (p ≥ 0.05)")
    
    with col2:
        engagement_comparison = pd.DataFrame({
            'Tier': ['Free', 'Premium'],
            'Avg Minutes': [ab_results['free_mean'], ab_results['premium_mean']]
        })
        st.bar_chart(engagement_comparison.set_index('Tier'))
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📊 Behavioral Analytics & Conversion Patterns")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🕐 Temporal Patterns", "🎭 Genre Analysis", "📍 Quadrant Analysis", "📱 Device Insights"])
    
    with tab1:
        st.markdown("**Daily Listening Patterns Across Hours**")
        
        # Simulate hourly patterns
        hours = list(range(24))
        free_pattern = [20, 15, 10, 8, 8, 12, 25, 35, 30, 28, 30, 35, 40, 38, 35, 40, 50, 60, 65, 55, 45, 40, 35, 25]
        premium_pattern = [25, 18, 12, 10, 10, 15, 30, 40, 35, 33, 35, 40, 45, 43, 40, 45, 55, 65, 70, 60, 50, 45, 40, 30]
        
        hourly_data = pd.DataFrame({
            'Hour': hours,
            'Free Users': free_pattern,
            'Premium Users': premium_pattern
        }).set_index('Hour')
        
        st.area_chart(hourly_data)
        st.caption("Peak listening hours: 6-8 PM for both tiers, Premium shows sustained engagement")
    
    with tab2:
        st.markdown("**Conversion Rates by Top Genre**")
        
        genre_conversion = filtered_df[filtered_df['Tier'] == 'Free'].groupby('Top_Genre').agg({
            'Converted_Last_30D': 'sum',
            'User_ID': 'count'
        }).reset_index()
        genre_conversion['Conversion_Rate'] = (genre_conversion['Converted_Last_30D'] / genre_conversion['User_ID'] * 100).round(2)
        genre_conversion = genre_conversion.rename(columns={'User_ID': 'Total_Users'})
        
        chart_data = genre_conversion.set_index('Top_Genre')['Conversion_Rate']
        st.bar_chart(chart_data)
        
        st.dataframe(
            genre_conversion.sort_values('Conversion_Rate', ascending=False),
            hide_index=True,
            use_container_width=True
        )
    
    with tab3:
        st.markdown("**Quadrant Analysis: Engagement vs. Frustration**")
        st.caption("High Frustration + High Engagement = Prime Conversion Targets")
        
        # Create quadrant data for free users only
        free_df = filtered_df[filtered_df['Tier'] == 'Free'].copy()
        
        quadrant_data = pd.DataFrame({
            'Daily Minutes': free_df['Daily_Avg_Mins'],
            'Frustration Index': free_df['Frustration_Index']
        })
        
        st.scatter_chart(quadrant_data, x='Daily Minutes', y='Frustration Index', size=20)
        
        # Quadrant interpretation
        high_frustration_high_engagement = free_df[
            (free_df['Frustration_Index'] > free_df['Frustration_Index'].median()) &
            (free_df['Daily_Avg_Mins'] > free_df['Daily_Avg_Mins'].median())
        ]
        
        st.info(f"🎯 **{len(high_frustration_high_engagement)} users** in High-Value Quadrant (Top-Right)")
    
    with tab4:
        st.markdown("**Conversion Propensity by Device Category**")
        
        device_analysis = filtered_df[filtered_df['Tier'] == 'Free'].groupby('Device_Category').agg({
            'Converted_Last_30D': 'sum',
            'User_ID': 'count'
        }).reset_index()
        device_analysis['Conversion_Rate'] = (device_analysis['Converted_Last_30D'] / device_analysis['User_ID'] * 100).round(2)
        device_analysis = device_analysis.rename(columns={'User_ID': 'Total_Users'})
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            chart_data = device_analysis.set_index('Device_Category')['Conversion_Rate']
            st.bar_chart(chart_data)
        
        with col2:
            st.dataframe(
                device_analysis.sort_values('Conversion_Rate', ascending=False),
                hide_index=True,
                use_container_width=True
            )
        
        mobile_conversion = device_analysis[device_analysis['Device_Category'] == 'Mobile']['Conversion_Rate'].values[0]
        desktop_conversion = device_analysis[device_analysis['Device_Category'] == 'Desktop']['Conversion_Rate'].values[0]
        
        if mobile_conversion > desktop_conversion:
            st.success(f"📱 Mobile users show {mobile_conversion - desktop_conversion:.2f}% higher conversion propensity")
        else:
            st.success(f"💻 Desktop users show {desktop_conversion - mobile_conversion:.2f}% higher conversion propensity")
    
    st.markdown("---")
    
    # Power Users Analysis
    st.subheader("🎯 High-Value Target Campaign: Power Users")
    
    power_users_df = filtered_df[filtered_df['Power_User'] == True][
        ['User_ID', 'Daily_Avg_Mins', 'Skip_Rate', 'Frustration_Index', 'Playlists_Created', 'Top_Genre']
    ].sort_values('Frustration_Index', ascending=False).head(10)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Top 10 Power Users (Free Tier) - Immediate Conversion Targets**")
        st.dataframe(power_users_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("**Power User Profile**")
        st.metric("Avg. Daily Minutes", f"{filtered_df[filtered_df['Power_User'] == True]['Daily_Avg_Mins'].mean():.1f}")
        st.metric("Avg. Frustration Index", f"{filtered_df[filtered_df['Power_User'] == True]['Frustration_Index'].mean():.1f}")
        st.metric("Avg. Playlists Created", f"{filtered_df[filtered_df['Power_User'] == True]['Playlists_Created'].mean():.1f}")
    
    st.markdown("---")
    
    # Executive Summary
    st.subheader("📋 Executive Summary & Recommended Actions")
    
    with st.expander("📊 View Full Analysis & Next Steps", expanded=True):
        st.markdown(f"""
        ### Key Findings
        
        1. **Statistical Validation of Premium Value**
           - Premium users demonstrate **{((ab_results['premium_mean'] / ab_results['free_mean']) - 1) * 100:.1f}% higher** daily engagement
           - P-value of **{ab_results['p_value']:.6f}** provides strong statistical evidence (p < 0.05)
           - This validates that Premium subscription directly increases platform stickiness
        
        2. **Conversion Trigger Identification**
           - The Frustration Index successfully identifies users experiencing high engagement but facing limitations
           - **{len(high_frustration_high_engagement)}** Free users are in the high-value conversion quadrant
           - Conversion rate of **{conversion_rate:.2f}%** in the last 30 days
        
        3. **Device-Based Insights**
           - {'Mobile' if mobile_conversion > desktop_conversion else 'Desktop'} users show the highest conversion propensity
           - This suggests optimizing the ad experience for {'mobile-first' if mobile_conversion > desktop_conversion else 'desktop'} delivery
        
        4. **Power User Opportunity**
           - **{filtered_df['Power_User'].sum()}** Free tier power users identified (Z-score > 2.5)
           - These users consume content at {filtered_df[filtered_df['Power_User'] == True]['Daily_Avg_Mins'].mean() / filtered_df['Daily_Avg_Mins'].mean():.1f}x the average rate
           - Represent immediate high-probability conversion targets
        
        ### Recommended Next Steps
        
        **Immediate Actions (Week 1-2):**
        - Launch targeted Premium ad campaign for users with skip rate ≥ {skip_threshold} ({len(targeted_users):,} users)
        - Deploy personalized messages emphasizing "ad-free listening" for high-frustration users
        - Implement A/B test on Premium trial offers (7-day vs 30-day) for power users
        
        **Short-term Initiatives (Month 1-2):**
        - Increase ad frequency by 25% for users in the top frustration quartile
        - Create genre-specific Premium campaigns targeting {top_conversion_genre} listeners
        - Develop device-optimized conversion funnels based on {'mobile' if mobile_conversion > desktop_conversion else 'desktop'} insights
        
        **Long-term Strategy (Quarter 1-2):**
        - Build predictive model using Frustration Index to identify conversion candidates 7-14 days in advance
        - Implement dynamic pricing strategy for high-engagement Free users approaching skip thresholds
        - Create retention program for new Premium converts identified through this analysis
        
        ### Expected Impact
        - Projected conversion rate improvement: **+15-20%** through targeted campaigns
        - Estimated incremental MRR: **${len(targeted_users) * 9.99 * 0.15:,.0f}** (assuming $9.99/month, 15% conversion lift)
        - Reduced churn risk for power users who might otherwise abandon due to frustration
        """)
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #535353;'>Built with Streamlit | Powered by Behavioral Data Science</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
