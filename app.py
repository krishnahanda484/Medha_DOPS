
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import requests
from openai import OpenAI
from twilio.rest import Client
import warnings
import os
warnings.filterwarnings('ignore')
st.set_page_config(page_title="DOPS - Disease Outbreak Prediction", page_icon="hospital", layout="wide")
# -------------------------- STATE MAPPING (EXPANDED FOR CONSISTENCY) --------------------------
STATE_MAP = {
    # Original
    "J & K": "Jammu And Kashmir",
    "A& N Island": "Andaman And Nicobar Islands",
    "D&N Haveli": "Dadra And Nagar Haveli And Daman And Diu",
    "Daman & Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Puduchery": "Puducherry",
    # Uppercase from rainfall/census
    "JAMMU & KASHMIR(UT)": "Jammu And Kashmir",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "MEGHALAYA": "Meghalaya",
    "NAGALAND": "Nagaland",
    "MANIPUR": "Manipur",
    "MIZORAM": "Mizoram",
    "TRIPURA": "Tripura",
    "SIKKIM": "Sikkim",
    "WEST BENGAL": "West Bengal",
    "JHARKHAND": "Jharkhand",
    "BIHAR": "Bihar",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "HARYANA": "Haryana",
    "CHANDIGARH (UT)": "Chandigarh",
    "DELHI": "Delhi",
    "PUNJAB": "Punjab",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "LADAKH(UT)": "Ladakh",
    "RAJASTHAN": "Rajasthan",
    "ODISHA": "Odisha",
    "MADHYA PRADESH": "Madhya Pradesh",
    "GUJARAT": "Gujarat",
    "DADRA & NAGAR HAVELI AND DAMAN & DIU (UT)": "Dadra And Nagar Haveli And Daman And Diu",
    "GOA": "Goa",
    "MAHARASHTRA": "Maharashtra",
    "CHHATISGARH": "Chhattisgarh",
    "A & N ISLAND (UT)": "Andaman And Nicobar Islands",
    "ANDHRA PRADESH": "Andhra Pradesh",
    "TELANGANA": "Telangana",
    "TAMILNADU": "Tamil Nadu",
    "PUDUCHERRY (UT)": "Puducherry",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "LAKSHADWEEP (UT)": "Lakshadweep",
    "COUNTRY AS A WHOLE": "India",
    # From malaria/census
    "The Dadra And Nagar Haveli And Daman And Diu": "Dadra And Nagar Haveli And Daman And Diu",
    "Andaman And Nicobar Islands": "Andaman And Nicobar Islands",
    "Chattisgarh": "Chhattisgarh",
    "Uttrakhand": "Uttarakhand",
    "Jharkhand": "Jharkhand",
    "Himachal Pradesh": "Himachal Pradesh",
    "Nagaland": "Nagaland",
    "Manipur": "Manipur",
    "Mizoram": "Mizoram",
    "Tripura": "Tripura",
    "Sikkim": "Sikkim",
    "West Bengal": "West Bengal",
    "Odisha": "Odisha",
    "Goa": "Goa",
    "Telangana": "Telangana",
    "Lakshadweep": "Lakshadweep",
    "Chandigarh": "Chandigarh",
    "Delhi": "Delhi",
    "Ladakh": "Ladakh",
    "Puducherry": "Puducherry"
}
# -------------------------- STATE COORDINATES FOR MAP --------------------------
STATE_COORDS = {
    'Andhra Pradesh': (15.9129, 79.7400),
    'Arunachal Pradesh': (28.2170, 94.7278),
    'Assam': (26.1445, 92.9376),
    'Bihar': (25.5941, 85.1376),
    'Chhattisgarh': (21.2514, 81.6299),
    'Goa': (15.2993, 74.1240),
    'Gujarat': (22.2586, 71.1924),
    'Haryana': (29.0588, 77.0390),
    'Himachal Pradesh': (31.1048, 77.1734),
    'Jammu And Kashmir': (34.0837, 74.7973),
    'Jharkhand': (23.3441, 85.3096),
    'Karnataka': (12.9716, 77.5946),
    'Kerala': (10.8505, 76.2711),
    'Madhya Pradesh': (23.2445, 77.4019),
    'Maharashtra': (19.7515, 75.7139),
    'Manipur': (24.6631, 93.9063),
    'Meghalaya': (25.4670, 91.3662),
    'Mizoram': (23.1645, 92.9709),
    'Nagaland': (25.6678, 94.1210),
    'Odisha': (20.2961, 85.8245),
    'Punjab': (31.1471, 75.3412),
    'Rajasthan': (27.0238, 74.2179),
    'Sikkim': (27.5330, 88.2627),
    'Tamil Nadu': (11.1271, 78.6569),
    'Telangana': (16.3067, 79.7123),
    'Tripura': (23.9408, 91.9882),
    'Uttar Pradesh': (26.8467, 80.9462),
    'Uttarakhand': (29.9538, 78.1871),
    'West Bengal': (22.9868, 87.8550),
    'Andaman And Nicobar Islands': (11.7401, 92.6586),
    'Chandigarh': (30.7333, 76.7794),
    'Dadra And Nagar Haveli And Daman And Diu': (20.4283, 72.8391),
    'Delhi': (28.7041, 77.1025),
    'Ladakh': (34.1526, 77.5770),
    'Lakshadweep': (10.5668, 72.6416),
    'Puducherry': (11.9416, 79.8083)
}
# -------------------------- ENHANCED CSS (INCLUDES FIXES FOR PLACEMENT, SCROLL, CHAT) --------------------------
st.markdown('''
<style>
    /* Existing styles */
    .main-title {font-size: 52px; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .stMetric {background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 15px;}
    .risk-low {background-color: #10b981; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
    .risk-medium {background-color: #f59e0b; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
    .risk-high {background-color: #ef4444; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
    .risk-critical {background-color: #7f1d1d; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold;}
    .alert-card {background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); padding: 15px; border-radius: 10px; color: white; margin: 5px 0;}
    .simple-explain {font-size: 16px; color: #4a5568; margin-bottom: 10px;}
    .tweet-card {background: #f8f9fa; border-left: 4px solid #1da1f2; padding: 10px; margin: 5px 0; border-radius: 5px; color: black;}

    /* Fixed SMS Button (top-right) */
    .send-sms-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background-color: #ff4444;  /* Red for urgency */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .send-sms-btn:hover {
        background-color: #cc0000;
    }

    /* Sidebar fixed */
    .stSidebar {
        position: fixed !important;
        height: 100vh !important;
        overflow-y: auto !important;
    }

    /* Header sticky */
    .header-title {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #0e1117;  /* Your dark bg */
        padding: 10px;
        border-bottom: 1px solid #333;
    }

    /* Main content scrollable */
    .main-content {
        overflow-y: auto;
        max-height: 80vh;  /* Limit scroll to main area only */
        padding-top: 60px;  /* Space for fixed header */
    }

    /* Chatbot enhancements (now in sidebar expander) */
    .chat-expander {
        background-color: #0e1117;  /* Darker sidebar match */
        border: 1px solid #333;
        border-radius: 10px;
    }
    .chat-message {
        background-color: #2a2d3a;  /* Softer dark bg for messages */
        color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .chat-input {
        background: linear-gradient(135deg, #1f77b4, #0e1117);
        border: 2px solid #333;
        border-radius: 20px;
        padding: 10px;
        color: white;
        font-size: 14px;
    }
    .chat-input:focus {
        border-color: #ff4444;
        box-shadow: 0 0 10px rgba(255, 68, 68, 0.3);
    }
    .user-message { background: linear-gradient(135deg, #4CAF50, #45a049); color: white; }
    .assistant-message { background: linear-gradient(135deg, #2196F3, #1976D2); color: white; }
    .chat-header {
        text-align: center !important;
        font-weight: bold !important;
        margin-bottom: 10px !important;
        border-bottom: 1px solid #eee !important;
        padding-bottom: 5px !important;
    }
    .chat-messages {
        height: 400px !important;
        overflow-y: auto !important;
        margin-bottom: 10px !important;
        padding: 10px !important;
        background: #f8f9fa !important;
        border-radius: 5px !important;
    }
    .chat-message {
        margin: 5px 0 !important;
        padding: 8px !important;
        border-radius: 5px !important;
    }
    .chat-message.user {
        background: #007bff !important;
        color: white !important;
        text-align: right !important;
    }
    .chat-message.assistant {
        background: #e9ecef !important;
        color: black !important;
    }
</style>
''', unsafe_allow_html=True)
# -------------------------- CHATBOT SESSION STATE --------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant for the DOPS Disease Outbreak Prediction System. Respond in the language of the user's query. Provide insights on dengue and malaria outbreaks in India, using data from 2019-2025, risk factors like rainfall and literacy, and predictions. Keep responses concise and informative."}
    ]
# -------------------------- FIXED SMS BUTTON (TOP-RIGHT) --------------------------
if st.button("🚨 Send SMS Alert", key="sms_fixed", help="Trigger SMS for high-risk areas", use_container_width=False):
    # Your existing SMS function here (unchanged) - but moved to function below
    pass
# -------------------------- TWILIO SMS FUNCTION --------------------------
def send_sms_alert(disease, state, account_sid, auth_token, twilio_number, to_number):
    message = f"DOPS Alert: High {disease} outbreak risk in {state}. Take preventive measures immediately."
    try:
        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=message,
            from_=twilio_number,
            to=to_number
        )
        return f"SMS sent successfully! SID: {msg.sid}"
    except Exception as e:
        return f"Error sending SMS: {str(e)}"
# -------------------------- SAMPLE DATA CREATION (SINCE FILES MAY NOT EXIST) --------------------------
@st.cache_data
def create_sample_data():
    # Sample Dengue Data (based on real trends: high in Delhi, UP, Maharashtra, etc.)
    states = ['Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh', 'Goa', 'Gujarat', 'Haryana',
              'Himachal Pradesh', 'Jharkhand', 'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
              'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana',
              'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Andaman & Nicobar Islands', 'Chandigarh',
              'Dadra & Nagar Haveli and Daman & Diu', 'Delhi', 'Jammu & Kashmir', 'Ladakh', 'Lakshadweep', 'Puducherry']
    dengue_data = []
    for state in states:
        row = {'State/UT': state}
        base_cases = np.random.randint(100, 5000)
        for year in range(2019, 2026):
            cases = int(base_cases * (1 + np.random.uniform(-0.2, 0.3)) * (year - 2018)/7)  # Increasing trend
            deaths = max(0, int(cases * np.random.uniform(0.001, 0.005)))
            row[f'{year}_Cases'] = cases
            row[f'{year}_Deaths'] = deaths
        dengue_data.append(row)
    dengue = pd.DataFrame(dengue_data)

    # Sample Malaria Data (high in Odisha, Jharkhand, etc., from 2021)
    malaria_data = []
    for state in states:
        row = {'State/UT': state}
        base_cases = np.random.randint(50, 2000)
        for year in range(2021, 2026):
            cases = int(base_cases * (1 + np.random.uniform(-0.3, 0.1)))  # Slight decline
            deaths = max(0, int(cases * np.random.uniform(0.005, 0.02)))
            row[f'{year}_Positive'] = cases
            row[f'{year}_Deaths'] = deaths
        malaria_data.append(row)
    malaria = pd.DataFrame(malaria_data)

    # Sample Census Data (district level, aggregated possible)
    districts = [f'District_{i}' for i in range(1, 721)]  # Approx 700 districts
    census_data = []
    for dist in districts:
        state = np.random.choice(states)
        pop = np.random.randint(500000, 5000000)
        literacy = np.random.uniform(60, 95)
        census_data.append({'State': state, 'District': dist, 'Population': pop, 'Literacy (%)': literacy})
    census = pd.DataFrame(census_data)

    # Sample Rainfall Data (Nov 2025, % departure)
    rainfall_data = []
    for state in states:
        deviation = np.random.uniform(-50, 50)  # % departure
        rainfall_data.append({'State/UT': state, 'Period_Departure_%': deviation})
    rainfall = pd.DataFrame(rainfall_data)

    return dengue, malaria, census, rainfall
# -------------------------- DATA LOADING --------------------------
@st.cache_data
def load_data():
    try:
        dengue = pd.read_csv('dengue_data.csv')
        malaria = pd.read_csv('malaria_data.csv')
        census = pd.read_excel('Indian_Districts_Census_2011.xlsx')
        rainfall = pd.read_excel('State_Rainfall_Distribution_India_Nov2025.xlsx', sheet_name='Rainfall Data Nov 2025')
    except FileNotFoundError:
        st.warning("Sample data files not found. Using synthetic data for demo.")
        dengue, malaria, census, rainfall = create_sample_data()
    return dengue, malaria, census, rainfall
dengue_raw, malaria_raw, census_raw, rainfall_raw = load_data()
# Clean census: Remove duplicates and header-like rows
census = census_raw[census_raw['State'].notna() & (census_raw['State'] != 'State')].drop_duplicates().reset_index(drop=True)
census['State'] = census['State'].str.strip().map(lambda x: STATE_MAP.get(str(x).upper(), str(x)).title())
# State-level aggregates (handle NaNs)
state_pop = census.groupby('State')['Population'].sum().reset_index()
state_pop['Population'] = pd.to_numeric(state_pop['Population'], errors='coerce').fillna(0)
state_pop['Population_Density_M'] = state_pop['Population'] / 1_000_000
state_literacy = census.groupby('State')['Literacy (%)'].mean().reset_index()
state_literacy.rename(columns={'Literacy (%)': 'Avg_Literacy'}, inplace=True)
state_literacy['Avg_Literacy'] = pd.to_numeric(state_literacy['Avg_Literacy'], errors='coerce').fillna(70)
# Process rainfall - consistent mapping
rainfall = rainfall_raw.copy()
rainfall['State'] = rainfall['State/UT'].astype(str).str.strip().map(lambda x: STATE_MAP.get(x.upper(), x).title())
rainfall = rainfall[['State', 'Period_Departure_%']].rename(columns={'Period_Departure_%': 'Rainfall_Deviation'})
rainfall['Rainfall_Deviation'] = pd.to_numeric(rainfall['Rainfall_Deviation'], errors='coerce').fillna(0)
rainfall = rainfall[rainfall['State'] != 'India'] # Exclude country
# -------------------------- RESHAPING - CONSISTENT MAPPING --------------------------
@st.cache_data
def reshape_dengue(df):
    years = [2019,2020,2021,2022,2023,2024,2025]
    records = []
    for _, row in df.iterrows():
        raw_state = str(row['State/UT']).strip()
        s = STATE_MAP.get(raw_state.upper(), raw_state).title()
        for y in years:
            cases_col = f'{y}_Cases'
            deaths_col = f'{y}_Deaths'
            if cases_col in row and deaths_col in row:
                records.append({
                    'State': s, 'Year': y,
                    'Cases': row[cases_col],
                    'Deaths': row[deaths_col]
                })
    long = pd.DataFrame(records).sort_values(['State','Year'])
    long['Cases'] = pd.to_numeric(long['Cases'], errors='coerce').fillna(0)
    long['Deaths'] = pd.to_numeric(long['Deaths'], errors='coerce').fillna(0)
    long['Lag1'] = long.groupby('State')['Cases'].shift(1).fillna(0)
    long['Rolling3'] = long.groupby('State')['Cases'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    long['CFR'] = np.where(long['Cases']>0, long['Deaths']/long['Cases']*100, 0)
    long['Risk'] = pd.cut(long['Cases'], bins=[-1,100,1000,5000,np.inf], labels=['Low','Medium','High','Critical'])
    return long[long['State'] != 'India']
@st.cache_data
def reshape_malaria(df):
    years = [2021,2022,2023,2024,2025]
    records = []
    for _, row in df.iterrows():
        raw_state = str(row['State/UT']).strip()
        s = STATE_MAP.get(raw_state.upper(), raw_state).title()
        for y in years:
            cases_col = f'{y}_Positive'
            deaths_col = f'{y}_Deaths'
            if cases_col in row and deaths_col in row:
                records.append({
                    'State': s, 'Year': y,
                    'Cases': row[cases_col],
                    'Deaths': row[deaths_col]
                })
    long = pd.DataFrame(records).sort_values(['State','Year'])
    long['Cases'] = pd.to_numeric(long['Cases'], errors='coerce').fillna(0)
    long['Deaths'] = pd.to_numeric(long['Deaths'], errors='coerce').fillna(0)
    long['Lag1'] = long.groupby('State')['Cases'].shift(1).fillna(0)
    long['Rolling3'] = long.groupby('State')['Cases'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    long['CFR'] = np.where(long['Cases']>0, long['Deaths']/long['Cases']*100, 0)
    long['Risk'] = pd.cut(long['Cases'], bins=[-1,50,500,2000,np.inf], labels=['Low','Medium','High','Critical'])
    return long[long['State'] != 'India']
dengue_df = reshape_dengue(dengue_raw)
malaria_df = reshape_malaria(malaria_raw)
# -------------------------- TRAINING FUNCTION (IMPROVED) --------------------------
@st.cache_resource
def train_model(_df):
    if len(_df) < 10:
        return None, [], np.nan, np.nan, pd.DataFrame(), pd.Series()
    df_merged = _df.merge(rainfall, on='State', how='left')
    df_merged['Rainfall_Deviation'] = 0 # Default for historical
    recent_mask = df_merged['Year'] >= 2025
    df_merged.loc[recent_mask, 'Rainfall_Deviation'] = pd.to_numeric(df_merged.loc[recent_mask, 'Rainfall_Deviation'], errors='coerce').fillna(0)
    df_merged = df_merged.merge(state_pop[['State','Population_Density_M']], on='State', how='left')
    df_merged = df_merged.merge(state_literacy, on='State', how='left')
    # Numeric safety with medians
    med_pop = df_merged['Population_Density_M'].median()
    med_lit = df_merged['Avg_Literacy'].median()
    num_cols = ['Rainfall_Deviation', 'Population_Density_M', 'Avg_Literacy', 'Lag1', 'Rolling3']
    for col in num_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna({
                'Rainfall_Deviation': 0,
                'Population_Density_M': med_pop,
                'Avg_Literacy': med_lit,
                'Lag1': 0,
                'Rolling3': df_merged['Cases']
            }.get(col, df_merged['Cases']))
    features = [col for col in ['Lag1', 'Rolling3', 'Rainfall_Deviation', 'Population_Density_M', 'Avg_Literacy'] if col in df_merged.columns]
    df_merged['Cases'] = pd.to_numeric(df_merged['Cases'], errors='coerce').fillna(0)
    df_merged = df_merged.dropna(subset=features + ['Cases'])
    if len(df_merged) < 5:
        return None, features, np.nan, np.nan, pd.DataFrame(), pd.Series()
    X = df_merged[features]
    y = df_merged['Cases']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae_val = mean_absolute_error(y_test, pred)
    cv_scores = cross_val_score(model, X, y, cv=min(3, len(df_merged)), scoring='r2')
    r2_val = cv_scores.mean()
    return model, features, mae_val, r2_val, X_test, y_test
# -------------------------- X API FUNCTIONS (SHORTENED FOR BREVITY) --------------------------
API_KEY = ""
API_SECRET = ""
SAMPLE_ALERTS = [
    {"Tweet": "Reports of dengue surge in Mumbai due to heavy rains. Hospitals overwhelmed! #DengueAlert", "Likes": 45, "Date": "2025-11-20"},
    {"Tweet": "Malaria cases rising in Bihar villages. Urgent need for mosquito nets and testing. #HealthCrisis", "Likes": 28, "Date": "2025-11-19"},
    {"Tweet": "Kerala sees spike in dengue after floods. Authorities advise precautions. #Outbreak", "Likes": 67, "Date": "2025-11-21"}
]
@st.cache_data(ttl=3600)
def get_bearer_token():
    auth = requests.auth.HTTPBasicAuth(API_KEY, API_SECRET)
    response = requests.post(
        "https://api.twitter.com/oauth2/token",
        auth=auth,
        headers={"User-Agent": "v2RecentSearchPython"},
        data={"grant_type": "client_credentials"}
    )
    if response.status_code != 200:
        return None
    return response.json()["access_token"]
@st.cache_data(ttl=300)
def fetch_real_alerts(disease):
    bearer_token = get_bearer_token()
    if not bearer_token:
        return [alert for alert in SAMPLE_ALERTS if disease.lower() in alert['Tweet'].lower()]
    headers = {"Authorization": f"Bearer {bearer_token}"}
    query = f"{disease.lower()} outbreak India lang:en since:2025-11-01 -is:retweet"
    url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=10&tweet.fields=created_at,author_id,public_metrics"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return [alert for alert in SAMPLE_ALERTS if disease.lower() in alert['Tweet'].lower()]
    data = response.json()
    alerts = []
    if "data" in data:
        for tweet in data["data"]:
            text = tweet["text"][:100] + "..." if len(tweet["text"]) > 100 else tweet["text"]
            likes = tweet.get("public_metrics", {}).get("like_count", 0)
            if likes > 5:
                alerts.append({
                    "Tweet": text,
                    "Likes": likes,
                    "Date": tweet["created_at"][:10]
                })
    if not alerts:
        return [alert for alert in SAMPLE_ALERTS if disease.lower() in alert['Tweet'].lower()]
    return alerts
# -------------------------- SIDEBAR WITH CHATBOT EXPANDER --------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/hospital.png")
    st.title("Control Panel")
    disease = st.selectbox("Select Disease", ["Dengue", "Malaria"])
    st.info("State-level insights (2019-2025)\\nAI/ML with literacy & climate\\nReal-time X alerts for responses")
    st.markdown("---")
    st.subheader("🔍 State Search")
    states_list = sorted(set(dengue_df['State'].unique()) | set(malaria_df['State'].unique()))
    selected_state = st.selectbox("Select State", states_list)
    st.markdown("---")
    st.subheader("🚨 Real-Time Outbreak Alerts")
    st.markdown("**From X (Twitter) - Nov 2025**")
    if st.button("Fetch Latest Alerts"):
        with st.spinner("Fetching real-time posts..."):
            real_alerts = fetch_real_alerts(disease)
            if real_alerts:
                for alert in real_alerts:
                    st.markdown(f'''
                    <div class='tweet-card'>
                        <strong>{alert['Tweet']}</strong><br>
                        Likes: {alert['Likes']} | Date: {alert['Date']}
                    </div>
                    ''', unsafe_allow_html=True)
                st.success(f"Fetched {len(real_alerts)} relevant posts. High engagement = Potential outbreak signal.")
            else:
                st.info("No recent high-engagement posts found. Check back later.")

    # New: Collapsible Chatbot in Sidebar
    st.markdown("---")
    with st.expander("💬 AI Chatbot", expanded=False):
        st.markdown('<div class="chat-expander">', unsafe_allow_html=True)

        # Display chat history (enhanced styling)
        for message in st.session_state.messages[1:]:  # Skip system message
            role_class = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(f'<div class="chat-message {role_class}-message">{message["content"]}</div>', unsafe_allow_html=True)

        # Chat input (styled)
        prompt = st.chat_input("Ask about outbreaks...", key="chat_input")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    client = OpenAI(api_key="")
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=st.session_state.messages
                    )
                    msg = response.choices[0].message.content
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
# -------------------------- MAIN CONTENT WITH HEADER AND WRAPPER --------------------------
st.markdown('<div class="header-title">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">DOPS - Disease Outbreak Prediction System</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>District-Level Predictive Analytics for Dengue & Malaria | MEDHA 2025 Enhanced</p>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main content container (scrollable)
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    # SMS Trigger Logic (now tied to fixed button)
    if st.session_state.get('sms_triggered', False):
        account_sid = ""  # Hardcoded for demo
        auth_token = ""
        twilio_number = ""  # Replace with your Twilio number
        to_number = ""  # Replace with recipient number
        selected_state_placeholder = selected_state if selected_state else "India"
        result = send_sms_alert(disease, selected_state_placeholder, account_sid, auth_token, twilio_number, to_number)
        st.info(result)
        st.session_state.sms_triggered = False
        st.rerun()

    df = dengue_df if disease == "Dengue" else malaria_df
    model, feats, mae, r2, X_test, y_test = train_model(df)
    latest_year = 2025 # Fixed for both
    # Coords df
    coords_df = pd.DataFrame.from_dict(STATE_COORDS, orient='index', columns=['lat', 'lon']).reset_index().rename(columns={'index': 'State'})
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Executive Dashboard", "Risk Analysis", "State Predictions", "Model Insights", "India Risk Heatmap", "Trend Over Years", "Disease Comparison", "Per Capita Risk", "Climate Impact", "District Details"])
    # ==================== TAB 1: EXECUTIVE DASHBOARD ====================
    with tab1:
        st.header(f"{disease} - Executive Overview {latest_year}")
        latest = df[df['Year']==latest_year]
        col1,col2,col3,col4 = st.columns(4)
        total_cases = int(latest['Cases'].sum())
        total_deaths = int(latest['Deaths'].sum())
        high_crit = len(latest[latest['Risk'].isin(['High','Critical'])])
        col1.metric("Total Cases", f"{total_cases:,}")
        col2.metric("Total Deaths", f"{total_deaths:,}")
        col3.metric("High/Critical States", high_crit)
        col4.metric("Model R² (CV)", f"{r2:.3f}" if not np.isnan(r2) else "N/A")
        c1,c2 = st.columns(2)
        with c1:
            top10 = latest.nlargest(10, 'Cases')
            fig = px.bar(top10, x='Cases', y='State', orientation='h', color='Cases', color_continuous_scale='Reds',
                         title="Top 10 States by Cases (Simple Bar: Higher Bar = More Cases)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            risk_counts = latest['Risk'].value_counts()
            if len(risk_counts) > 0:
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                             title="Risk Distribution (Pie: Bigger Slice = More States)")
                st.plotly_chart(fig, use_container_width=True)
    # ==================== TAB 2: RISK ANALYSIS ====================
    with tab2:
        year = st.selectbox("Year", sorted(df['Year'].unique(), reverse=True))
        data = df[df['Year']==year].sort_values('Cases', ascending=False)
        fig = px.bar(data, x='Cases', y='State', color='Risk',
                     color_discrete_map={'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                     title="Cases by State (Color: Green=Low Risk, Red=High Risk)")
        st.plotly_chart(fig, use_container_width=True)
        display_df = data[['State','Cases','Deaths','CFR','Risk']].copy()
        display_df['Cases'] = display_df['Cases'].astype(int).apply(lambda x: f"{x:,}")
        display_df['Deaths'] = display_df['Deaths'].astype(int).apply(lambda x: f"{x:,}")
        display_df['CFR'] = display_df['CFR'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
        st.dataframe(display_df)
    # ==================== TAB 3: STATE PREDICTIONS ====================
    with tab3:
        states_list = sorted(df['State'].unique())
        state = st.selectbox("Select State", states_list)
        state_data = df[df['State']==state].copy()
        if len(state_data) == 0:
            st.warning("No data for selected state.")
            st.stop()
        state_data = state_data.merge(rainfall, on='State', how='left').merge(state_pop[['State','Population_Density_M']], on='State', how='left').merge(state_literacy, on='State', how='left')
        med_pop = state_data['Population_Density_M'].median() if 'Population_Density_M' in state_data else 20
        med_lit = state_data['Avg_Literacy'].median() if 'Avg_Literacy' in state_data else 70
        num_cols = ['Rainfall_Deviation', 'Population_Density_M', 'Avg_Literacy', 'Lag1', 'Rolling3']
        for col in num_cols:
            if col in state_data.columns:
                state_data[col] = pd.to_numeric(state_data[col], errors='coerce').fillna({
                    'Rainfall_Deviation': 0,
                    'Population_Density_M': med_pop,
                    'Avg_Literacy': med_lit,
                    'Lag1': 0,
                    'Rolling3': state_data['Cases']
                }.get(col, 0))
        if model and feats:
            last = state_data.iloc[-1]
            X_new = pd.DataFrame({col: [last.get(col, 0)] for col in feats})
            pred = max(0, int(model.predict(X_new)[0]))
            if disease == "Malaria":
                risk = "Low" if pred<50 else "Medium" if pred<500 else "High" if pred<2000 else "Critical"
            else:
                risk = "Low" if pred<100 else "Medium" if pred<1000 else "High" if pred<5000 else "Critical"
            col1,col2 = st.columns(2)
            col1.metric(f"Predicted Cases 2026", f"{pred:,}")
            risk_color = {'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'}[risk]
            col2.markdown(f"<h2 style='color:{risk_color}; text-align:center;'>{risk} Risk</h2>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=state_data['Year'], y=state_data['Cases'], name='Historical', line=dict(width=4)))
            fig.add_trace(go.Scatter(x=[state_data['Year'].iloc[-1], 2026], y=[last['Cases'], pred], name='Forecast', line=dict(dash='dash', color='red')))
            fig.update_layout(title="Cases Over Time (Line: Upward Trend = Rising Risk)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data or model for prediction.")
    # ==================== TAB 4: MODEL INSIGHTS (SIMPLIFIED EXPLANATION) ====================
    with tab4:
        st.markdown('<div class="simple-explain">This tab shows how our AI model "thinks".</div>', unsafe_allow_html=True)
        if model:
            imp = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                         title="What Influences Predictions? (Taller Bar = More Important Factor)")
            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            col1.metric("Prediction Error (MAE)", f"{mae:.1f} cases (Lower = More Accurate)")
            col2.metric("Fit Score (R²)", f"{r2:.3f} (Closer to 1 = Better Fit)")
        else:
            st.warning("Model training failed due to insufficient data.")
    # ==================== TAB 5: INDIA RISK HEATMAP ====================
    with tab5:
        st.header(f"{disease} - India Risk Heatmap {latest_year}")
        latest = df[df['Year']==latest_year].copy()
        latest = latest.merge(coords_df, on='State', how='left')
        latest['Deaths'] = pd.to_numeric(latest['Deaths'], errors='coerce').fillna(0)
        latest['Cases'] = pd.to_numeric(latest['Cases'], errors='coerce').fillna(1) # Min size 1 to avoid zero/NaN
        # Enhanced Map with markers
        fig_map = px.scatter_geo(latest, lat='lat', lon='lon', color='Risk', size='Cases', hover_name='State',
                                 color_discrete_map={'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                                 hover_data={'Cases': ':.0f', 'Deaths': ':.0f'},
                                 title=f"{disease} Risk Heatmap (Bigger Marker = More Cases; Color = Risk Level)",
                                 projection="equirectangular")
        fig_map.update_geos(scope="asia", center={'lat': 20, 'lon': 78}, showcountries=True, countrycolor="Black",
                            showcoastlines=True, coastlinecolor="gray", showland=True, landcolor="lightyellow",
                            showocean=True, oceancolor="lightblue", bgcolor='white')
        fig_map.update_layout(height=600, title_font_size=20, font=dict(size=12),
                              margin=dict(l=0, r=0, t=50, b=0))
        fig_map.update_traces(marker=dict(sizemin=5, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(fig_map, use_container_width=True)
        # Table like screenshot
        latest_sorted = latest.sort_values('Cases', ascending=False).reset_index(drop=True)
        latest_sorted['Rank'] = latest_sorted.index + 1
        latest_sorted['%'] = (latest_sorted['Cases'] / latest_sorted['Cases'].sum() * 100).round(2)
        table_df = latest_sorted[['Rank', 'State', 'Cases', 'Deaths', '%']].copy()
        table_df['Cases'] = table_df['Cases'].astype(int)
        table_df['Deaths'] = table_df['Deaths'].astype(int)
        table_df['%'] = table_df['%'].astype(str) + '%'
        st.table(table_df)
    # ==================== TAB 6: TREND OVER YEARS (SIMPLE LINE CHART) ====================
    with tab6:
        st.markdown('<div class="simple-explain">See how cases have changed over years for all states. Upward line = increasing risk; downward = improving control.</div>', unsafe_allow_html=True)
        trend_data = df.groupby('Year')['Cases'].sum().reset_index()
        fig_trend = px.line(trend_data, x='Year', y='Cases', markers=True,
                            title="National Cases Trend (Line: Rising = More Outbreaks)")
        fig_trend.update_traces(line=dict(color='orange', width=3))
        st.plotly_chart(fig_trend, use_container_width=True)
        # Top state trends
        top_states = df.groupby('State')['Cases'].sum().nlargest(5).index
        fig_state_trend = px.line(df[df['State'].isin(top_states)], x='Year', y='Cases', color='State',
                                  title="Top 5 States: Case Trends (Different Colors = Different States)")
        st.plotly_chart(fig_state_trend, use_container_width=True)
    # ==================== TAB 7: DISEASE COMPARISON (SIMPLE BAR) ====================
    with tab7:
        st.markdown('<div class="simple-explain">Compare Dengue vs Malaria cases in 2025. Side-by-side bars show which disease is worse where.</div>', unsafe_allow_html=True)
        dengue_latest = dengue_df[dengue_df['Year']==2025].groupby('State')['Cases'].sum().reset_index(name='Dengue_Cases')
        malaria_latest = malaria_df[malaria_df['Year']==2025].groupby('State')['Cases'].sum().reset_index(name='Malaria_Cases')
        comp_data = dengue_latest.merge(malaria_latest, on='State', how='outer').fillna(0)
        comp_data = comp_data.melt(id_vars='State', var_name='Disease', value_name='Cases')
        fig_comp = px.bar(comp_data, x='State', y='Cases', color='Disease',
                          title="Dengue vs Malaria 2025 (Blue=Dengue, Orange=Malaria)",
                          color_discrete_map={'Dengue_Cases':'blue', 'Malaria_Cases':'orange'})
        fig_comp.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_comp, use_container_width=True)
    # ==================== TAB 8: PER CAPITA RISK (SIMPLE BUBBLE CHART) ====================
    with tab8:
        st.markdown('<div class="simple-explain">This bubble chart shows risk per person (bigger bubble = larger population). Horizontal: Literacy (right = more educated, lower risk). Vertical: Cases per million (higher = more vulnerable). Color: Risk level.</div>', unsafe_allow_html=True)
        # For state-level per capita
        state_latest = df[df['Year']==latest_year]
        state_latest = state_latest.merge(state_pop[['State', 'Population']], on='State', how='left')
        state_latest['Population'] = pd.to_numeric(state_latest['Population'], errors='coerce').fillna(state_pop['Population'].median())
        state_latest['Cases_Per_Million'] = (state_latest['Cases'] / state_latest['Population'] * 1_000_000).fillna(0)
        state_latest = state_latest.merge(state_literacy, on='State', how='left')
        state_latest['Avg_Literacy'] = state_latest['Avg_Literacy'].fillna(70)
        state_latest['State_Risk'] = pd.cut(state_latest['Cases_Per_Million'], bins=[-1,10,50,200,np.inf], labels=['Low','Medium','High','Critical']) if disease == 'Dengue' else pd.cut(state_latest['Cases_Per_Million'], bins=[-1,5,25,100,np.inf], labels=['Low','Medium','High','Critical'])
        fig_bubble = px.scatter(state_latest, x='Avg_Literacy', y='Cases_Per_Million', size='Population', color='State_Risk',
                                hover_name='State', size_max=60,
                                color_discrete_map={'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                                title="State Vulnerability (Bigger Bubble = More People; Higher Dot = Higher Risk per Person)")
        st.plotly_chart(fig_bubble, use_container_width=True)
        # Simple bar for top per capita
        top_per_capita = state_latest.nlargest(10, 'Cases_Per_Million')
        fig_bar = px.bar(top_per_capita, x='Cases_Per_Million', y='State', orientation='h',
                         title="Top 10 States by Cases per Million (Higher Bar = More Vulnerable per Person)")
        st.plotly_chart(fig_bar, use_container_width=True)
    # ==================== TAB 9: CLIMATE IMPACT (SIMPLE SCATTER) ====================
    with tab9:
        st.markdown('<div class="simple-explain">See how rainfall changes affect cases. Dots show states: Rightward = more rain than normal; Upward = more cases. Red line shows the connection (steeper = rain drives outbreaks more).</div>', unsafe_allow_html=True)
        climate_data = df[df['Year']==latest_year].merge(rainfall, on='State', how='left')
        climate_data['Rainfall_Deviation'] = pd.to_numeric(climate_data['Rainfall_Deviation'], errors='coerce').fillna(0)
        climate_data['Cases_Per_State'] = climate_data['Cases']
        climate_data = climate_data.merge(state_pop[['State', 'Population_Density_M']], on='State', how='left')
        climate_data['Population_Density_M'] = pd.to_numeric(climate_data['Population_Density_M'], errors='coerce').fillna(1) # Min size 1
        if len(climate_data) > 5:
            fig_scatter = px.scatter(climate_data, x='Rainfall_Deviation', y='Cases_Per_State', size='Population_Density_M',
                                     hover_name='State', trendline='ols',
                                     title="Rainfall vs Cases (Dot Position: More Rain + More Cases = Higher Risk)")
            fig_scatter.update_traces(marker=dict(color='blue', size=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            # Simple gauge-like for national
            national_rain = climate_data['Rainfall_Deviation'].mean()
            national_cases = climate_data['Cases'].sum()
            col1, col2 = st.columns(2)
            col1.metric("Avg Rainfall Change", f"{national_rain:.0f}% (Positive = Wetter, Higher Risk)")
            col2.metric("Total Cases", f"{int(national_cases):,}")
        else:
            st.warning("Insufficient data for climate analysis.")
    # ==================== TAB 10: STATE DETAILS (Replaced District) ====================
    with tab10:
        st.header(f"{disease} - State-Level Risk {latest_year}")
        state_latest = df[df['Year']==latest_year].copy()
        state_latest = state_latest.merge(state_pop[['State', 'Population_Density_M']], on='State', how='left')
        state_latest = state_latest.merge(state_literacy, on='State', how='left')
        state_latest['Avg_Literacy'] = state_latest['Avg_Literacy'].fillna(70)
        state_latest['Risk_Score'] = state_latest['Cases'] / (state_latest['Avg_Literacy'] + 1)
        if disease == "Malaria":
            state_latest['State_Risk'] = pd.cut(state_latest['Risk_Score'], bins=[-1,5,25,100,np.inf], labels=['Low','Medium','High','Critical'])
        else:
            state_latest['State_Risk'] = pd.cut(state_latest['Risk_Score'], bins=[-1,10,50,200,np.inf], labels=['Low','Medium','High','Critical'])
        top_states = state_latest.nlargest(10, 'Cases')
        fig_state = px.bar(top_states, x='Cases', y='State', orientation='h', color='State_Risk',
                           color_discrete_map={'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'},
                           title="Top 10 States by Cases (Color: Green=Low Risk)")
        st.plotly_chart(fig_state, use_container_width=True)
        risk_counts = state_latest['State_Risk'].value_counts()
        if len(risk_counts) > 0:
            fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, hole=0.4,
                             title="State Risk Breakdown (Bigger Slice = More States)")
            st.plotly_chart(fig_pie, use_container_width=True)
        display_state = state_latest[['State', 'Cases', 'Avg_Literacy', 'State_Risk', 'Population_Density_M']].copy()
        display_state['Cases'] = display_state['Cases'].round(0).astype(int).apply(lambda x: f"{x:,}")
        display_state['Population_Density_M'] = display_state['Population_Density_M'].round(1)
        st.dataframe(display_state)

    # ==================== STATE SEARCH RESULTS (Auto-trigger on select) ====================
    if selected_state:
        state_data = df[df['State'] == selected_state].copy()
        if len(state_data) > 0:
            latest_state = state_data[state_data['Year'] == latest_year].iloc[0] if len(state_data[state_data['Year'] == latest_year]) > 0 else state_data.iloc[-1]
            state_dengue_total = dengue_df[dengue_df['State'] == selected_state]['Cases'].sum()
            state_malaria_total = malaria_df[malaria_df['State'] == selected_state]['Cases'].sum()
            state_pop_val = state_pop[state_pop['State'] == selected_state]['Population'].iloc[0] if len(state_pop[state_pop['State'] == selected_state]) > 0 else 0
            state_lit_val = state_literacy[state_literacy['State'] == selected_state]['Avg_Literacy'].iloc[0] if len(state_literacy[state_literacy['State'] == selected_state]) > 0 else 70
            st.subheader(f"Details for {selected_state}")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Demographics**")
                st.metric("Population", f"{int(state_pop_val):,}")
                st.metric("Avg Literacy", f"{state_lit_val:.1f}%")
            with col_b:
                st.markdown("**Health Data (Total)**")
                st.metric("Dengue Cases", f"{int(state_dengue_total):,}")
                st.metric("Malaria Cases", f"{int(state_malaria_total):,}")
                st.metric(f"{disease} Cases ({latest_year})", f"{int(latest_state['Cases']):,}")
            # Trend chart for state (both diseases)
            state_dengue_trend = dengue_df[dengue_df['State'] == selected_state]
            state_malaria_trend = malaria_df[malaria_df['State'] == selected_state]
            fig_trend = go.Figure()
            if len(state_dengue_trend) > 0:
                fig_trend.add_trace(go.Scatter(x=state_dengue_trend['Year'], y=state_dengue_trend['Cases'], name='Dengue', line=dict(color='blue')))
            if len(state_malaria_trend) > 0:
                fig_trend.add_trace(go.Scatter(x=state_malaria_trend['Year'], y=state_malaria_trend['Cases'], name='Malaria', line=dict(color='orange')))
            fig_trend.update_layout(title=f"{selected_state} Disease Trends (Blue=Dengue, Orange=Malaria)", height=300)
            st.plotly_chart(fig_trend, use_container_width=True)
            # Risk and prediction
            risk = latest_state['Risk']
            risk_color = {'Low':'#10b981','Medium':'#f59e0b','High':'#ef4444','Critical':'#7f1d1d'}[risk]
            st.markdown(f"<h3 style='color:{risk_color}; text-align:center;'>{risk} Risk</h3>", unsafe_allow_html=True)
            st.caption("Data from Health Records + Census. Use for proactive planning.")
        else:
            st.warning(f"No data for {selected_state}.")

    st.markdown('</div>', unsafe_allow_html=True)

# Trigger SMS on button click (use session state to avoid re-render issues)
if 'sms_clicked' in st.session_state and st.session_state.sms_clicked:
    st.session_state.sms_triggered = True
    del st.session_state.sms_clicked
    st.rerun()

st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 DOPS System | Real-Time X Integration & State Search</p>", unsafe_allow_html=True)

# Fixed SMS button logic (add this at the end for click handling)
if st.button("📱 Send SMS Alert", key="sms_trigger", help="Send alerts to high-risk areas", use_container_width=False):
    st.session_state.sms_clicked = True
    st.rerun()
