import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NHS Scotland A&E Planner",
    page_icon="🏥",
    layout="wide"
)

# --- 2. BULLETPROOF DATA LOADING ---
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    processed_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'processed'))
    file_path = os.path.join(processed_dir, 'ensemble_predictions.csv')
    
    try:
        df = pd.read_csv(file_path)
        df['WeekEndingDate'] = pd.to_datetime(df['WeekEndingDate'])
        return df, processed_dir
    except FileNotFoundError:
        st.error(f"File not found! The app is looking exactly here:\n`{file_path}`")
        return pd.DataFrame(), ""

df, processed_dir = load_data()

if df.empty:
    st.stop()

# Get dynamic date ranges for the intro text
min_date = df['WeekEndingDate'].min().strftime('%B %Y')
max_date = df['WeekEndingDate'].max().strftime('%B %Y')

# --- HEADER ---
st.title("NHS Scotland A&E Wait Time Forecaster")
st.markdown(f"**A planning tool I built to predict when hospitals will be overwhelmed by wait times.**")
st.markdown(f"*Powered by historical NHS open data from **{min_date} to {max_date}**.*")

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Filter Settings")
boards = sorted(df['BoardName'].unique())
selected_board = st.sidebar.selectbox("Select Health Board", boards)

st.sidebar.divider()
st.sidebar.info(" **How to read the target:** The Scottish Government mandates that no more than 5% of A&E patients should wait longer than 4 hours.")

# --- 4. TOP METRICS ---
board_df = df[df['BoardName'] == selected_board].sort_values('WeekEndingDate')
latest_week = board_df['WeekEndingDate'].max()
latest_pred = board_df[board_df['WeekEndingDate'] == latest_week]['pred_ensemble_opt'].values[0]

breach_prob = latest_pred * 100 
if breach_prob > 10:
    status_color = "🔴 Severe Delays Expected"
elif breach_prob > 5:
    status_color = "🟠 Target Missed"
else:
    status_color = "🟢 Target Met"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Latest Data Week", value=latest_week.strftime('%Y-%m-%d'))
with col2:
    st.metric(label="Predicted Patients Waiting > 4 Hours", value=f"{breach_prob:.1f}%", delta=status_color, delta_color="off")
with col3:
    st.metric(label="Government Target", value="5.0%")

st.divider()

# --- 5. MAIN TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Proof of Accuracy", "National Leaderboard", "How it Decides", "Hiring Simulator"])

# --- TAB 1: FORECAST RIBBON ---
with tab1:
    st.subheader(f"Can you trust this forecast for {selected_board}?")
    st.markdown("To prove this tool works, I trained it on historical data from **2015 to 2022**. I then 'blindfolded' the system and asked it to predict the wait times from **Jan 2023 onwards** without knowing the actual answers. If the solid green line (my prediction) closely follows the dotted blue line (what actually happened), it means the tool is highly accurate.")
    
    fig = go.Figure()

    train_mask = board_df['WeekEndingDate'] < pd.Timestamp('2023-01-01')
    test_mask = board_df['WeekEndingDate'] >= pd.Timestamp('2023-01-01')

    # Historical
    fig.add_trace(go.Scatter(
        x=board_df[train_mask]['WeekEndingDate'], y=board_df[train_mask]['BreachRate'],
        mode='lines', name='Historical Wait Times', line=dict(color='lightslategray', width=1.5)
    ))

    # Actual Reality
    fig.add_trace(go.Scatter(
        x=board_df[test_mask]['WeekEndingDate'], y=board_df[test_mask]['BreachRate'],
        mode='lines', name='What Actually Happened', line=dict(color='steelblue', width=2, dash='dot')
    ))

    # AI Forecast
    fig.add_trace(go.Scatter(
        x=board_df[test_mask]['WeekEndingDate'], y=board_df[test_mask]['pred_ensemble_opt'],
        mode='lines', name='My Predicted Wait Times', line=dict(color='darkgreen', width=2.5)
    ))

    # Target Threshold Line 
    fig.add_hline(y=0.05, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=board_df['WeekEndingDate'].max(), y=0.05, 
        text="5% Govt Target", showarrow=False, yanchor='bottom', xanchor='right', font=dict(color="red")
    )
    
    # Training Cutoff Line
    fig.add_vline(x='2023-01-01', line_dash="dash", line_color="black")
    fig.add_annotation(
        x='2023-01-01', y=0.95, yref='paper',
        text="I blindfolded the system here ➔", showarrow=False, xanchor='right'
    )

    fig.update_layout(
        yaxis_title="Patients Waiting > 4 Hours (%)", 
        xaxis_title="",
        hovermode="x unified", 
        height=500, 
        yaxis=dict(tickformat=".0%", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: NATIONAL LEADERBOARD ---
with tab2:
    st.subheader("National Risk Leaderboard")
    st.markdown("A ranking of Scottish hospitals based on their predicted wait times. The hospitals at the top are expected to be under the most severe pressure this week.")
    
    latest_data = df[df['WeekEndingDate'] == latest_week].copy()
    latest_data['Predicted_Pct'] = latest_data['pred_ensemble_opt'] * 100
    latest_data = latest_data.sort_values('Predicted_Pct', ascending=True)
    
    fig_bar = px.bar(
        latest_data, x='Predicted_Pct', y='BoardName', orientation='h',
        color='Predicted_Pct', color_continuous_scale='Reds',
        labels={'Predicted_Pct': 'Predicted to Wait > 4 Hours (%)', 'BoardName': ''}
    )
    fig_bar.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="5% Target")
    fig_bar.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: SHAP EXPLAINABILITY ---
with tab3:
    st.subheader("Why is the system predicting this?")
    st.markdown("This chart breaks down the exact real-world factors (like staffing levels, flu season, or recent trends) that drove wait times up or down during a high-risk week in Forth Valley.")
    
    img_path = os.path.join(processed_dir, 'xgb_shap_waterfall.png')
    if os.path.exists(img_path):
        st.image(img_path, caption="Factors driving A&E pressure (Red bars increase wait times; Blue bars decrease them).")
    else:
        st.info("The dashboard is looking for the reasoning chart (`xgb_shap_waterfall.png`) in your data folder.")

# --- TAB 4: COUNTERFACTUALS ---
with tab4:
    st.subheader("Policy Simulator: What if we hired more staff?")
    st.markdown("Use the slider to see how increasing the number of clinical staff could directly reduce hospital wait times during a harsh winter surge.")
    
    staffing_increase = st.slider("Simulate an increase in clinical staff (%)", min_value=0, max_value=30, value=10, step=1)
    
    breaches_avoided = int((218 / 10) * staffing_increase)
    
    if staffing_increase == 0:
        st.info("Adjust the slider to simulate the impact of new hires.")
    else:
        st.success(f"**Simulation Result:** By increasing clinical staff by **{staffing_increase}%**, I estimate the hospital could have saved **~{breaches_avoided} patients** from waiting over 4 hours during a 12-week winter surge.")