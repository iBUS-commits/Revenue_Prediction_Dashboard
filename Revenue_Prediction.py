import streamlit as st

import pandas as pd
import numpy as np
import warnings
from io import StringIO
import pdfplumber
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Suppress warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Set page config for better appearance
st.set_page_config(
    page_title="IBUS Financial & Climate Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# IBUS Website Colors based on the logo
IBUS_PRIMARY = "#110361"  # Light blue from logo (was dark blue)
IBUS_SECONDARY = "#097BA8"  # IBUS secondary blue
IBUS_ACCENT = "#ff0015"  # IBUS accent orange
IBUS_LIGHT = "#f0f2f5"  # Light background
IBUS_DARK = "#043362"  # Dark text
IBUS_SUCCESS = "#28a745"  # Success green
IBUS_DANGER = "#dc3545"  # Danger red

# Custom CSS for IBUS styling
st.markdown(f"""
    <style>
        /* Main app styling */
        .stApp {{
            background-color: {IBUS_LIGHT};
        }}
        
        /* Headers with underlines */
        h1, h2, h3, h4, h5, h6 {{
            color: {IBUS_PRIMARY} !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            border-bottom: 2px solid {IBUS_PRIMARY};
            padding-bottom: 8px;
            margin-bottom: 16px;
        }}
        
        /* Sidebar headers - no underline */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] h4, 
        [data-testid="stSidebar"] h5, 
        [data-testid="stSidebar"] h6 {{
            border-bottom: none;
            padding-bottom: 0;
            margin-bottom: 8px;
        }}
        
        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {IBUS_PRIMARY}, {IBUS_SECONDARY});
            color: white;
        }}
        [data-testid="stSidebar"] .stRadio label, 
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stSlider label {{
            color: white !important;
            font-weight: 600;
        }}
        
        /* Forecast Configuration specific styling */
        [data-testid="stSidebar"] .stNumberInput[data-testid*="forecast_years"] label,
        [data-testid="stSidebar"] h3:contains("Forecast Configuration") {{
            color: white !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
        }}
                                                 
        /* Sidebar markdown text */
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown p {{
            color: white !important;
            font-weight: 600;
        }}
        
        /* Input text in sidebar */
        [data-testid="stSidebar"] input[type="text"],
        [data-testid="stSidebar"] input[type="number"],
        [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {{
            color: {IBUS_DARK} !important;
            font-weight: 500;
        }}
        
        /* Financial parameters text */
        .financial-parameter {{
            color: white !important;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        
        /* Buttons */
        .stButton>button {{
            background-color: {IBUS_ACCENT};
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: {IBUS_PRIMARY};
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Input widgets */
        .stSelectbox, .stSlider, .stNumberInput, .stTextInput, .stTextArea {{
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        /* Dataframes */
        .stDataFrame {{
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        /* Tabs */
        .st-b7 {{
            background-color: {IBUS_LIGHT};
        }}
        [data-baseweb="tab-list"] {{
            gap: 10px;
        }}
        [data-baseweb="tab"] {{
            background-color: {IBUS_LIGHT};
            border-radius: 8px !important;
            padding: 10px 20px !important;
            margin-right: 10px !important;
            transition: all 0.3s;
        }}
        [data-baseweb="tab"]:hover {{
            background-color: #e9ecef;
        }}
        [aria-selected="true"] {{
            background-color: {IBUS_PRIMARY} !important;
            color: white !important;
        }}
        
        /* Expanders */
        .stExpander {{
            border-radius: 8px !important;
            border: 1px solid #dee2e6 !important;
        }}
        .st-expanderHeader {{
            font-weight: 600 !important;
            color: {IBUS_PRIMARY} !important;
        }}
        
        /* IBUS Header */
        .ibus-header {{
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
        }}
        .ibus-logo {{
            height: 50px;
            margin-right: 15px;
        }}
        .ibus-title {{
            color: {IBUS_PRIMARY};
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
        }}
        .ibus-subtitle {{
            color: {IBUS_DARK};
            font-size: 1rem;
            margin: 0;
        }}
        
        /* Metrics */
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .metric-title {{
            color: {IBUS_DARK};
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .metric-value {{
            color: {IBUS_PRIMARY};
            font-size: 1.8rem;
            font-weight: 700;
        }}
        .metric-change {{
            font-size: 0.9rem;
            font-weight: 500;
        }}
        .positive {{
            color: {IBUS_SUCCESS};
        }}
        .negative {{
            color: {IBUS_DANGER};
        }}
    </style>
""", unsafe_allow_html=True)

# IBUS Header with logo
st.markdown(f"""
    <h1 style="color: {IBUS_PRIMARY}; margin-bottom: 16px; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px;">Financial & Climate Forecast Dashboard</h1>
    <p style="color: {IBUS_PRIMARY}; font-size: 1.2rem; margin-top: 0;">Multi-year forecasting with integrated financial and climate analytics</p>
""", unsafe_allow_html=True)

# Country and state data
COUNTRIES = {
    "Australia": ["New South Wales", "Queensland", "Victoria", "Western Australia"],
    "Brazil": ["Minas Gerais", "Rio de Janeiro", "Rio Grande do Sul", "São Paulo"],
    "Canada": ["Alberta", "British Columbia", "Ontario", "Quebec"],
    "China": ["Beijing", "Guangdong", "Shanghai", "Sichuan"],
    "France": ["Auvergne-Rhône-Alpes", "Île-de-France", "Provence-Alpes-Côte d'Azur"],
    "Germany": ["Bavaria", "Berlin", "Hamburg", "North Rhine-Westphalia"],
    "India": ["Delhi", "Karnataka", "Maharashtra", "Tamil Nadu"],
    "Japan": ["Hokkaido", "Kyoto", "Osaka", "Tokyo"],
    "United Kingdom": ["England", "Northern Ireland", "Scotland", "Wales"],
    "United States": ["California", "Florida", "Illinois", "New York", "Texas"]
}

@st.cache_resource
def load_model_and_data():
    # Create historical data with fiscal year labels
    current_year = datetime.now().year
    years = [f"FY{current_year-10+i}" for i in range(10)]
    
    data = pd.DataFrame({
        'FY': years,
        'R&D Spend': [165349.2, 162597.7, 153441.51, 144372.41, 142107.34,
                      131876.9, 134615.46, 130298.13, 120542.52, 123334.88],
        'Administration': [136897.8, 151377.59, 101145.55, 118671.85, 91391.77,
                           99814.71, 147198.87, 145530.06, 148718.95, 108679.17],
        'Marketing Spend': [471784.1, 443898.53, 407934.54, 383199.62, 366168.42,
                            362861.36, 127716.82, 323876.68, 311613.29, 304981.62],
        'Country': ['United States'] * 10,
        'State': ['New York', "California", "Florida", "New York", "Florida",
                  "New York", "California", "Florida", "New York", "California"],
        'Profit': [192261.83, 191792.06, 191050.39, 182901.99, 166187.94,
                   156991.12, 156122.51, 155752.6, 152211.77, 149759.96],
        'Rainfall': [120, 85, 95, 110, 100, 105, 80, 90, 115, 87],
        'Temperature': [30, 28, 29, 31, 27, 26, 25, 24, 30, 28]
    })

    # Create dummy variables for country and state
    data = pd.get_dummies(data, columns=['Country', 'State'], drop_first=True)

    n_lags = 3
    for lag in range(1, n_lags + 1):
        data[f'Rainfall_lag_{lag}'] = data['Rainfall'].shift(lag)
        data[f'Temperature_lag_{lag}'] = data['Temperature'].shift(lag)
    data.dropna(inplace=True)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    num_features = ['R&D Spend', 'Administration', 'Marketing Spend']
    poly_features = poly.fit_transform(data[num_features])
    poly_feature_names = poly.get_feature_names_out(num_features)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=data.index)

    data_final = pd.concat([poly_df, data.drop(columns=num_features)], axis=1)

    X = data_final.drop(columns=['Profit', 'Rainfall', 'Temperature', 'FY'])
    y = data_final[['Profit', 'Rainfall', 'Temperature']]

    scaler = StandardScaler()
    X_scaled_np = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled_np, columns=X.columns, index=X.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    base_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    param_dist = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20, 30],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2],
        'estimator__max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        base_model, param_dist, n_iter=10, cv=3,
        scoring='neg_mean_squared_error', verbose=0,
        random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    last_known = data[['Rainfall', 'Temperature']].iloc[-n_lags:].copy()
    last_known_rainfall = list(last_known['Rainfall'])
    last_known_temp = list(last_known['Temperature'])

    return best_model, scaler, poly, poly_feature_names, X.columns, n_lags, last_known_rainfall, last_known_temp, data

# Load model and data
best_model, scaler, poly, poly_feature_names, feature_cols, n_lags, last_known_rainfall, last_known_temp, historical_data = load_model_and_data()

# Initialize session state for dynamic year updates
current_year = datetime.now().year
if 'start_year' not in st.session_state:
    st.session_state.start_year = current_year - 5
if 'end_year' not in st.session_state:
    st.session_state.end_year = current_year
if 'forecast_years' not in st.session_state:
    st.session_state.forecast_years = 5
if 'chart_type' not in st.session_state:
    st.session_state.chart_type = 'Line'

def update_start_year():
    if st.session_state.start_year > st.session_state.end_year:
        st.session_state.end_year = st.session_state.start_year

def update_end_year():
    if st.session_state.end_year < st.session_state.start_year:
        st.session_state.start_year = st.session_state.end_year

# Sidebar configuration
with st.sidebar:
    # Add a white background to the logo
    st.markdown("""
        <style>
        [data-testid="stImage"] {
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Use the actual logo image
    st.image("ibus-logo.png", width=150)
    st.markdown('<h2 style="color:white;">Input Parameters</h2>', unsafe_allow_html=True)
    
    # Input method selection as tabs
    input_method = st.radio(
        "Select Input Method",
        ['Slider Input', 'Manual Entry', 'File Upload'],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_method == 'Slider Input':
        st.markdown('<h3 style="color:white;">Financial Parameters</h3>', unsafe_allow_html=True)
        
        # R&D Spend - just use the default slider with custom label
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Base R&D Spend ($)</p>', unsafe_allow_html=True)
        rnd_spend = st.slider('Base R&D Spend ($)', 50000, 200000, 140000, 1000, label_visibility="collapsed")
        
        # R&D Growth Rate
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">R&D Growth Rate (%/year)</p>', unsafe_allow_html=True)
        rnd_growth = st.slider('R&D Growth Rate (%/year)', 0.0, 10.0, 2.0, 0.1, label_visibility="collapsed")
        
        # Administration Spend
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Base Administration Spend ($)</p>', unsafe_allow_html=True)
        admin_spend = st.slider('Base Administration Spend ($)', 50000, 200000, 120000, 1000, label_visibility="collapsed")
        
        # Admin Growth Rate
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Administration Growth Rate (%/year)</p>', unsafe_allow_html=True)
        admin_growth = st.slider('Administration Growth Rate (%/year)', 0.0, 10.0, 1.0, 0.1, label_visibility="collapsed")
        
        # Marketing Spend
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Base Marketing Spend ($)</p>', unsafe_allow_html=True)
        marketing_spend = st.slider('Base Marketing Spend ($)', 50000, 500000, 350000, 1000, label_visibility="collapsed")
        
        # Marketing Growth Rate
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Marketing Growth Rate (%/year)</p>', unsafe_allow_html=True)
        marketing_growth = st.slider('Marketing Growth Rate (%/year)', 0.0, 10.0, 3.0, 0.1, label_visibility="collapsed")
        
        # Add this specific CSS for the Country and State dropdowns
        st.markdown("""
            <style>
            /* Target specifically the Country and State dropdowns */
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border: 2px solid #110361 !important;
            }
            
            /* Style the selected option text */
            div[data-baseweb="select"] span[aria-selected="true"] {
                color: #000000 !important;
                font-weight: bold !important;
            }
            
            /* Style the placeholder text */
            div[data-baseweb="select"] [data-testid="stSelectbox"] {
                color: #000000 !important;
                font-weight: bold !important;
            }
            
            /* Make sure the text is visible */
            .st-emotion-cache-1aehpvj, .st-emotion-cache-16idsys {
                color: #000000 !important;
                font-weight: bold !important;
            }
            
            /* Target any text inside the select box */
            div[data-baseweb="select"] * {
                color: #000000 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Then your country and state selectors with explicit labels
        st.markdown('<h3 style="color:white;">Location</h3>', unsafe_allow_html=True)
        country = st.selectbox('Country', list(COUNTRIES.keys()), format_func=lambda x: x)
        state = st.selectbox('State', COUNTRIES[country], format_func=lambda x: x)
        
        st.markdown('<h3 style="color:white;">Climate Parameters</h3>', unsafe_allow_html=True)
        
        # Base Rainfall - remove duplicate labels
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Base Rainfall (mm)</p>', unsafe_allow_html=True)
        base_rainfall = st.slider('Base Rainfall (mm)', 50, 500, 100, 5, label_visibility="collapsed")
        
        # Rainfall Trend - remove duplicate labels
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Rainfall Trend (mm/year)</p>', unsafe_allow_html=True)
        rainfall_trend = st.slider('Rainfall Trend (mm/year)', -50.0, 1000.0, 25.0, 0.1, label_visibility="collapsed")
        
        # Base Temperature - remove duplicate labels
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Base Temperature (°C)</p>', unsafe_allow_html=True)
        base_temp = st.slider('Base Temperature (°C)', 15, 35, 25, 1, label_visibility="collapsed")
        
        # Temperature Trend - remove duplicate labels
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Temperature Trend (°C/year)</p>', unsafe_allow_html=True)
        temp_trend = st.slider('Temperature Trend (°C/year)', -1.8, 1.8, 0.5, 0.1, label_visibility="collapsed")
        
        st.markdown('<h3 style="color:white; font-weight:bold; margin-bottom:5px;">Forecast Configuration</h3>', unsafe_allow_html=True)
        
        # Custom styling for the forecast years input field with more specific selectors
        st.markdown("""
            <style>
            /* Target the input field directly with higher specificity */
            div[data-testid="stNumberInput"][data-baseweb="input-spinner"] input,
            [data-testid="stNumberInput"] input[type="number"],
            [data-testid="stNumberInput"][data-testid*="forecast_years"] input {
                background-color: white !important;
                color: black !important;
                font-weight: bold !important;
                border: 2px solid white !important;
            }
            
            /* Target the container with higher specificity */
            div[data-testid="stNumberInput"][data-baseweb="input-spinner"],
            [data-testid="stNumberInput"][data-testid*="forecast_years"] {
                background-color: white !important;
                border-radius: 8px !important;
                padding: 2px !important;
            }
            
            /* Target the buttons with higher specificity */
            div[data-testid="stNumberInput"] button,
            [data-testid="stNumberInput"][data-testid*="forecast_years"] button {
                background-color: white !important;
                color: #110361 !important;
                border: 1px solid #110361 !important;
            }
            
            /* Target the button icons with higher specificity */
            div[data-testid="stNumberInput"] button svg,
            [data-testid="stNumberInput"][data-testid*="forecast_years"] button svg {
                fill: #110361 !important;
            }
            
            /* Force override with !important on all properties */
            [data-testid="stNumberInput"] * {
                background-color: white !important;
            }
            
            [data-testid="stNumberInput"] input {
                color: black !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Custom label with strong styling
        st.markdown('<p style="color:white; font-weight:bold; margin-bottom:5px;">Number of Years to Predict:</p>', unsafe_allow_html=True)
        
        # Number input with hidden label
        forecast_years = st.number_input(
            'Number of Years to Predict', 
            min_value=1, 
            max_value=20, 
            key='forecast_years',
            label_visibility="collapsed"  # Hide the default label
        )
        
        # For all input methods, ensure we create a continuous range of years
        if input_method == 'Slider Input' or input_method == 'Manual Entry' or input_method == 'File Upload':
            # Get the last year from historical data
            if 'historical_data' in locals() and 'FY' in historical_data.columns:
                try:
                    # Extract the numeric part of the FY string and convert to int
                    last_historical_year = max([int(y[2:]) if isinstance(y, str) and y.startswith('FY') else int(y) 
                                              for y in historical_data['FY']])
                except:
                    last_historical_year = st.session_state.end_year
            else:
                last_historical_year = st.session_state.end_year
            
            # Create a continuous range from the last historical year + 1 to the forecast end
            forecast_range = list(range(last_historical_year + 1, last_historical_year + 1 + forecast_years))
            
            # Debug output to verify the range
            st.write(f"Debug: Forecast range from {last_historical_year + 1} to {last_historical_year + forecast_years}")
    
    elif input_method == 'Manual Entry':
        st.markdown('<h3 style="color:white;">Historical Data Range</h3>', unsafe_allow_html=True)

        # Add custom styling for the number input fields in Historical Data Range
        st.markdown("""
            <style>
            /* Target the number input fields */
            [data-testid="stNumberInput"] input {
                background-color: white !important;
                color: black !important;
                font-weight: bold !important;
                border: 2px solid #110361 !important;
            }
            
            /* Style the plus/minus buttons */
            [data-testid="stNumberInput"] button {
                background-color: white !important;
                color: #110361 !important;
                border: 1px solid #110361 !important;
            }
            
            /* Style the plus/minus button icons */
            [data-testid="stNumberInput"] button svg {
                fill: #110361 !important;
            }
            
            /* Style the container */
            [data-testid="stNumberInput"] {
                background-color: white !important;
                border-radius: 8px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Create a container for the year inputs with custom styling
        st.markdown("""
            <style>
            /* Style for the year input container */
            .year-inputs-container {
                display: flex;
                flex-direction: row;
                justify-content: space-between;
                align-items: flex-start;
                margin-bottom: 15px;
            }
            
            /* Style for each year input group */
            .year-input-group {
                width: 48%;
            }
            
            /* Style for year labels */
            .year-label {
                color: white;
                font-weight: bold;
                margin-bottom: 5px;
                display: block;
            }
            </style>
            
            <div class="year-inputs-container">
                <div class="year-input-group">
                    <span class="year-label">Start Year</span>
                </div>
                <div class="year-input-group">
                    <span class="year-label">End Year</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Create two columns for the inputs
        col1, col2 = st.columns(2)

        with col1:
            start_year = st.number_input(
                'Start Year',
                min_value=current_year-50,
                max_value=current_year+50,
                value=st.session_state.start_year,
                key='start_year',
                label_visibility="collapsed",  # Hide the default label
                on_change=update_start_year
            )

        with col2:
            end_year = st.number_input(
                'End Year',
                min_value=st.session_state.start_year,
                max_value=current_year+50,
                value=st.session_state.end_year,
                key='end_year',
                label_visibility="collapsed",  # Hide the default label
                on_change=update_end_year
            )

        # Country and state selection
        st.markdown('<h3 style="color:white;">Location</h3>', unsafe_allow_html=True)
        # Add specific styling for ONLY the Country and State dropdowns in Manual Entry
        st.markdown("""
            <style>
            /* Target only the select boxes, not the headings */
            [data-testid="stSelectbox"] div[data-baseweb="select"],
            [data-testid="stSelectbox"] div[role="combobox"] {
                background-color: white !important;
                border: 2px solid #110361 !important;
                border-radius: 8px !important;
            }
            
            /* Style the dropdown options */
            [data-testid="stSelectbox"] div[role="option"] {
                background-color: white !important;
                color: black !important;
            }
            
            /* Style only the text inside the select box */
            [data-testid="stSelectbox"] div[data-baseweb="select"] span,
            [data-testid="stSelectbox"] div[role="combobox"] span,
            [data-testid="stSelectbox"] div[data-baseweb="select"] div,
            [data-testid="stSelectbox"] div[role="combobox"] div {
                color: black !important;
                font-weight: bold !important;
            }
            
            /* Style the dropdown arrow */
            [data-testid="stSelectbox"] svg {
                color: black !important;
            }
            
            /* Force only the select box elements to have white background */
            [data-testid="stSelectbox"] div[data-baseweb="select"] *,
            [data-testid="stSelectbox"] div[role="combobox"] * {
                background-color: white !important;
            }
            
            /* Target all text elements inside the select box */
            [data-testid="stSelectbox"] div[data-baseweb="select"] *,
            [data-testid="stSelectbox"] div[role="combobox"] * {
                color: black !important;
            }
            
            /* Target the selected value text specifically */
            [data-testid="stSelectbox"] [aria-selected="true"],
            [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdown"] p {
                color: black !important;
                font-weight: bold !important;
            }
            
            /* Force only the number inputs to have white background */
            [data-testid="stNumberInput"] input {
                background-color: white !important;
                color: black !important;
                font-weight: bold !important;
                border: 2px solid #110361 !important;
            }
            
            /* Style the plus/minus buttons */
            [data-testid="stNumberInput"] button {
                background-color: white !important;
                color: #110361 !important;
                border: 1px solid #110361 !important;
            }
            
            /* Style the plus/minus button icons */
            [data-testid="stNumberInput"] button svg {
                fill: #110361 !important;
            }
            
            /* Style the container */
            [data-testid="stNumberInput"] {
                background-color: white !important;
                border-radius: 8px !important;
            }
            </style>
        """, unsafe_allow_html=True)
        country = st.selectbox('Country', list(COUNTRIES.keys()))
        state = st.selectbox('State', COUNTRIES[country])
        
        num_years = end_year - start_year + 1
        manual_data = []
        
        st.markdown('<h3 style="color:white;">Enter Historical Data</h3>', unsafe_allow_html=True)
        for i in range(num_years):
            year = start_year + i
            with st.expander(f"FY{year}"):
                cols = st.columns(3)
                with cols[0]:
                    profit = st.number_input(f'Profit ($)', value=150000 + i*5000, step=1000, key=f'profit_{year}')
                with cols[1]:
                    rnd_spend = st.number_input(f'R&D Spend ($)', value=140000 + i*1000, step=1000, key=f'rnd_{year}')
                with cols[2]:
                    admin_spend = st.number_input(f'Administration Spend ($)', value=120000 + i*1000, step=1000, key=f'admin_{year}')
                
                cols = st.columns(3)
                with cols[0]:
                    marketing_spend = st.number_input(f'Marketing Spend ($)', value=350000 + i*5000, step=1000, key=f'marketing_{year}')
                with cols[1]:
                    rainfall = st.number_input(f'Rainfall (mm)', value=100 + i*2, step=1, key=f'rain_{year}')
                with cols[2]:
                    temp = st.number_input(f'Temperature (°C)', value=25 + i, step=1, key=f'temp_{year}')
                
                manual_data.append([f'FY{year}', rnd_spend, admin_spend, marketing_spend, profit, rainfall, temp])
        
        st.markdown('<h3 style="color:white;">Forecast Configuration</h3>', unsafe_allow_html=True)

        # Add specific styling for the forecast years label in Manual Entry
        st.markdown("""
            <style>
            /* Target specifically the forecast years label in Manual Entry */
            [data-testid="stNumberInput"][data-testid*="forecast_years"] label,
            label:contains("Number of Years to Predict") {
                color: black !important;
                font-weight: bold !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Use a custom label with black text instead of relying on the default label
        st.markdown('<p style="color:black; font-weight:bold; margin-bottom:5px;">Number of Years to Predict:</p>', unsafe_allow_html=True)

        # Number input with hidden label
        forecast_years = st.number_input(
            'Number of Years to Predict', 
            min_value=1, 
            max_value=20, 
            key='forecast_years',
            label_visibility="collapsed"  # Hide the default label
        )
        
        # For all input methods, ensure we create a continuous range of years
        if input_method == 'Slider Input' or input_method == 'Manual Entry' or input_method == 'File Upload':
            # Get the last year from historical data
            if 'historical_data' in locals() and 'FY' in historical_data.columns:
                try:
                    # Extract the numeric part of the FY string and convert to int
                    last_historical_year = max([int(y[2:]) if isinstance(y, str) and y.startswith('FY') else int(y) 
                                              for y in historical_data['FY']])
                except:
                    last_historical_year = st.session_state.end_year
            else:
                last_historical_year = st.session_state.end_year
            
            # Create a continuous range from the last historical year + 1 to the forecast end
            forecast_range = list(range(last_historical_year + 1, last_historical_year + 1 + forecast_years))
            
            # Debug output to verify the range
            st.write(f"Debug: Forecast range from {last_historical_year + 1} to {last_historical_year + forecast_years}")
    
    elif input_method == 'File Upload':
        # Add styling specifically for ONLY the select boxes in the file upload section
        st.markdown("""
            <style>
            /* Target only the select boxes, not the headings */
            [data-testid="stSelectbox"] div[data-baseweb="select"],
            [data-testid="stSelectbox"] div[role="combobox"] {
                background-color: white !important;
                border: 2px solid #110361 !important;
                border-radius: 8px !important;
            }
            
            /* Style the dropdown options */
            [data-testid="stSelectbox"] div[role="option"] {
                background-color: white !important;
                color: black !important;
            }
            
            /* Style only the text inside the select box */
            [data-testid="stSelectbox"] div[data-baseweb="select"] span,
            [data-testid="stSelectbox"] div[role="combobox"] span,
            [data-testid="stSelectbox"] div[data-baseweb="select"] div,
            [data-testid="stSelectbox"] div[role="combobox"] div {
                color: black !important;
                font-weight: bold !important;
            }
            
            /* Style the dropdown arrow */
            [data-testid="stSelectbox"] svg {
                color: black !important;
            }
            
            /* Force only the select box elements to have white background */
            [data-testid="stSelectbox"] div[data-baseweb="select"] *,
            [data-testid="stSelectbox"] div[role="combobox"] * {
                background-color: white !important;
            }
            
            /* Target all text elements inside the select box */
            [data-testid="stSelectbox"] div[data-baseweb="select"] *,
            [data-testid="stSelectbox"] div[role="combobox"] * {
                color: black !important;
            }
            
            /* Target the selected value text specifically */
            [data-testid="stSelectbox"] [aria-selected="true"],
            [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdown"] p {
                color: black !important;
                font-weight: bold !important;
            }
            
            /* Force only the number inputs to have white background */
            [data-testid="stNumberInput"] input {
                background-color: white !important;
                color: black !important;
                font-weight: bold !important;
                border: 2px solid #110361 !important;
            }
            
            /* Style the plus/minus buttons */
            [data-testid="stNumberInput"] button {
                background-color: white !important;
                color: #110361 !important;
                border: 1px solid #110361 !important;
            }
            
            /* Style the plus/minus button icons */
            [data-testid="stNumberInput"] button svg {
                fill: #110361 !important;
            }
            
            /* Style the container */
            [data-testid="stNumberInput"] {
                background-color: white !important;
                border-radius: 8px !important;
            }
            
            /* Explicitly preserve heading styles */
            h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                /* Do not change these - this preserves the original styling */
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Use a completely different approach with stronger contrast
        st.markdown("""
            <div style="background-color: #110361; padding: 15px; border-radius: 10px; margin-bottom: 20px; border: 2px solid #0056b3;">
                <h3 style="color:white; font-size:1.2rem; margin:0;">Upload Historical Data</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Add a more visible file uploader with description
        st.write("Please upload your historical data file (supports various formats)")
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'xlsx', 'xls', 'json', 'pdf', 'txt', 'docx', 'xml', 'sas7bdat', 'parquet', 'feather'],
            help="Upload your historical data file containing financial and climate information"
        )
        
        if uploaded_file is not None:
            # Add styling for all select boxes and data tables in the file upload section
            st.markdown("""
                <style>
                /* Style for all select boxes in file upload section */
                [data-testid="stSelectbox"] div[data-baseweb="select"],
                [data-testid="stSelectbox"] div[role="combobox"] {
                    background-color: white !important;
                    border: 2px solid #110361 !important;
                    border-radius: 8px !important;
                }
                
                /* Style for text inside select boxes */
                [data-testid="stSelectbox"] div[data-baseweb="select"] span,
                [data-testid="stSelectbox"] div[role="combobox"] span,
                [data-testid="stSelectbox"] div[data-baseweb="select"] div,
                [data-testid="stSelectbox"] div[role="combobox"] div {
                    color: black !important;
                    font-weight: bold !important;
                }
                
                /* Style for dropdown options */
                [data-testid="stSelectbox"] div[role="option"] {
                    background-color: white !important;
                    color: black !important;
                }
                
                /* Style for data tables */
                .dataframe {
                    background-color: white !important;
                }
                
                .dataframe th, .dataframe td {
                    background-color: white !important;
                    color: black !important;
                    font-weight: normal !important;
                    border: 1px solid #ddd !important;
                }
                
                .dataframe th {
                    background-color: #f2f2f2 !important;
                    font-weight: bold !important;
                }
                
                /* Style for expanders that might contain tables */
                .st-emotion-cache-1r6slb0 {
                    background-color: white !important;
                }
                
                /* Style for multiselect boxes */
                [data-testid="stMultiSelect"] div[data-baseweb="select"],
                [data-testid="stMultiSelect"] div[role="combobox"] {
                    background-color: white !important;
                    border: 2px solid #110361 !important;
                    border-radius: 8px !important;
                }
                
                [data-testid="stMultiSelect"] div[data-baseweb="select"] span,
                [data-testid="stMultiSelect"] div[role="combobox"] span {
                    color: black !important;
                }
                
                /* Style for any text elements in the file upload section */
                .file-upload-section p, .file-upload-section label, .file-upload-section div {
                    color: white !important;
                }
                
                /* Style specifically for table headers in the file upload section */
                .file-upload-section .dataframe th {
                    background-color: #110361 !important;
                    color: white !important;
                    font-weight: bold !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Add a class to the file upload section for targeting
            st.markdown('<div class="file-upload-section">', unsafe_allow_html=True)
            
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                
                if file_extension == '.csv':
                    historical_data = pd.read_csv(uploaded_file)
                elif file_extension in ['.xlsx', '.xls']:
                    # Show sheet selection if multiple sheets exist
                    xls = pd.ExcelFile(uploaded_file)
                    if len(xls.sheet_names) > 1:
                        sheet_name = st.selectbox("Select sheet:", xls.sheet_names)
                    else:
                        sheet_name = xls.sheet_names[0]
                    historical_data = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                elif file_extension == '.json':
                    historical_data = pd.read_json(uploaded_file)
                elif file_extension == '.pdf':
                    st.info("Processing PDF file... This may take a moment.")
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Extract text
                    with pdfplumber.open(uploaded_file) as pdf:
                        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Extract tables
                    # Define extract_tables_from_pdf if not already defined
                    def extract_tables_from_pdf(pdf_file):
                        import pdfplumber
                        tables = []
                        with pdfplumber.open(pdf_file) as pdf:
                            for i, page in enumerate(pdf.pages):
                                page_tables = page.extract_tables()
                                for table in page_tables:
                                    # Convert to DataFrame and add to list
                                    df = pd.DataFrame(table)
                                    # Try to use first row as header if possible
                                    if df.shape[0] > 1:
                                        df.columns = df.iloc[0]
                                        df = df[1:]
                                    tables.append({'table': df, 'source': f'Page {i+1}'})
                        return tables

                    tables = extract_tables_from_pdf(uploaded_file)
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Extract graphs
                    def extract_graphs_from_pdf(pdf_file):
                        # Placeholder: actual graph extraction from PDF is not implemented
                        # Return an empty list to avoid errors
                        return []
                    graphs = extract_graphs_from_pdf(uploaded_file)
                    
                    # Display extraction results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"📊 Found {len(tables)} tables in the PDF")
                    with col2:
                        st.write(f"📈 Found {len(graphs)} potential graphs in the PDF")
                    
                    # Process tables
                    if tables:
                        # Let user select which table to use
                        table_options = [f"Table {i+1} ({t['source']}, {t['table'].shape[0]}x{t['table'].shape[1]})" 
                                         for i, t in enumerate(tables)]
                        selected_table_idx = st.selectbox("Select a table to use:", 
                                                         options=range(len(tables)),
                                                         format_func=lambda x: table_options[x])
                        
                        selected_table = tables[selected_table_idx]['table']
                        st.write("Selected table preview:")
                        st.dataframe(selected_table.head())
                        
                        # Basic data cleaning
                        # Remove completely empty rows and columns
                        selected_table = selected_table.dropna(how='all').dropna(axis=1, how='all')
                        
                        # Try to convert numeric columns
                        for col in selected_table.columns:
                            try:
                                selected_table[col] = pd.to_numeric(selected_table[col], errors='ignore')
                            except Exception:
                                pass
                        
                        # Offer choice between automatic and manual column mapping
                        mapping_method = st.radio(
                            "How would you like to map columns?",
                            ["Automatic (AI-based)", "Manual (select each column)"]
                        )
                        
                        if mapping_method == "Automatic (AI-based)":
                            # Use the intelligent mapping function
                            # Fallback: simple heuristic mapping for demonstration
                            def map_columns_intelligently(df):
                                """
                                Intelligently map columns from uploaded file to expected columns
                                using text similarity and content analysis.
                                """
                                st.info("Analyzing file structure and content...")
                                
                                # Expected features and their common synonyms/variations
                                expected_features = {
                                    'fy': ['fy', 'fiscal year', 'year', 'period', 'date', 'time'],
                                    'profit': ['profit', 'earnings', 'net income', 'income', 'revenue', 'net profit', 'gain'],
                                    'r&d_spend': ['r&d', 'research', 'development', 'r&d spend', 'research and development', 'r and d'],
                                    'administration': ['admin', 'administration', 'administrative', 'overhead', 'general', 'g&a'],
                                    'marketing_spend': ['marketing', 'advertising', 'promotion', 'sales', 'market'],
                                    'rainfall': ['rain', 'rainfall', 'precipitation', 'water', 'moisture'],
                                    'temperature': ['temperature', 'temp', 'average temperature', 'avg temp', 'mean temperature', 'climate']
                                }
                                
                                # Standard column names for output
                                standard_names = {
                                    'fy': 'FY',
                                    'profit': 'Profit',
                                    'r&d_spend': 'R&D Spend',
                                    'administration': 'Administration',
                                    'marketing_spend': 'Marketing Spend',
                                    'rainfall': 'Rainfall',
                                    'temperature': 'Temperature'
                                }
                                
                                # Ensure all column names are strings
                                df.columns = [str(col) for col in df.columns]
                                
                                # Clean column names
                                df.columns = df.columns.str.lower().str.strip()
                                
                                # Display original column names
                                st.markdown("<div style='color:#110361; font-weight:bold;'>Original columns:</div>", unsafe_allow_html=True)
                                st.markdown(f"<div style='color:#000000;'>{', '.join(df.columns.tolist())}</div>", unsafe_allow_html=True)
                                
                                # Step 1: Direct text matching
                                column_mapping = {}
                                for feature, synonyms in expected_features.items():
                                    for col in df.columns:
                                        # Check for exact match
                                        if col in synonyms:
                                            column_mapping[feature] = col
                                            break
                                        
                                        # Check for partial match (column contains the synonym or synonym contains column)
                                        for syn in synonyms:
                                            if syn in col or col in syn:
                                                column_mapping[feature] = col
                                                break
                                        
                                        if feature in column_mapping:
                                            break
                                
                                # Step 2: Fallback to heuristic matching if needed
                                if len(column_mapping) < len(expected_features):
                                    st.info("Heuristic matching for remaining columns...")
                                    for feature, synonyms in expected_features.items():
                                        if feature not in column_mapping:
                                            for col in df.columns:
                                                if any(syn in col for syn in synonyms):
                                                    column_mapping[feature] = col
                                                    break
                                
                                # Step 3: Assign default values for missing columns
                                for feature in expected_features:
                                    if feature not in column_mapping:
                                        st.warning(f"Could not map '{feature}'. Using default value.")
                                        column_mapping[feature] = None
                                
                                # Create a new DataFrame with mapped columns
                                mapped_df = pd.DataFrame(index=df.index)
                                for feature, mapped_col in column_mapping.items():
                                    if mapped_col is not None:
                                        mapped_df[standard_names[feature]] = df[mapped_col]
                                    else:
                                        # Use default values for missing columns
                                        if feature == 'fy':
                                            current_year = datetime.now().year
                                            mapped_df[standard_names[feature]] = [f"FY{current_year-len(df)+i}" for i in range(len(df))]
                                        elif feature == 'country':
                                            mapped_df[standard_names[feature]] = 'United States'
                                        elif feature == 'state':
                                            mapped_df[standard_names[feature]] = 'New York'
                                        elif feature in ['rainfall', 'temperature']:
                                            mapped_df[standard_names[feature]] = 100 if feature == 'rainfall' else 25
                                        else:
                                            mapped_df[standard_names[feature]] = 100000
                                
                                return mapped_df

                            try:
                                historical_data = map_columns_intelligently(selected_table)
                            except Exception as e:
                                st.error(f"Error in automatic mapping: {str(e)}")
                                # Fallback to a simpler approach
                                historical_data = pd.DataFrame(index=selected_table.index)
                                for req_col in ['FY', 'R&D Spend', 'Administration', 'Marketing Spend', 
                                                'Profit', 'Rainfall', 'Temperature', 'Country', 'State']:
                                    # Set default values
                                    if req_col == 'FY':
                                        current_year = datetime.now().year
                                        historical_data[req_col] = [f"FY{current_year-len(selected_table)+i}" for i in range(len(selected_table))]
                                    elif req_col == 'Country':
                                        historical_data[req_col] = 'United States'
                                    elif req_col == 'State':
                                        historical_data[req_col] = 'New York'
                                    elif req_col in ['Rainfall', 'Temperature']:
                                        historical_data[req_col] = 100 if req_col == 'Rainfall' else 25
                                    else:
                                        historical_data[req_col] = 100000
                        else:
                            # Manual column mapping
                            st.subheader("Manual Column Mapping")
                            st.info("Please select which column corresponds to each required field:")
                            
                            # Create a new DataFrame for the mapped data
                            historical_data = pd.DataFrame(index=selected_table.index)
                            
                            # Ensure all column names are strings
                            selected_table.columns = [str(col) for col in selected_table.columns]
                            
                            # For each required column, let the user select from available columns
                            col_options = ["None"] + list(selected_table.columns)
                            required_columns = ['FY', 'R&D Spend', 'Administration', 'Marketing Spend', 
                                               'Profit', 'Rainfall', 'Temperature', 'Country', 'State']
                            
                            for req_col in required_columns:
                                mapped_col = st.selectbox(
                                    f"Map '{req_col}' to:", 
                                    col_options,
                                    key=f"manual_map_{req_col}"
                                )
                                
                                if mapped_col != "None":
                                    historical_data[req_col] = selected_table[mapped_col]
                                else:
                                    # Use defaults for missing columns
                                    if req_col == 'FY':
                                        # Generate fiscal years
                                        current_year = datetime.now().year
                                        historical_data[req_col] = [f"FY{current_year-len(selected_table)+i}" for i in range(len(selected_table))]
                                    elif req_col == 'Country':
                                        historical_data[req_col] = 'United States'
                                    elif req_col == 'State':
                                        historical_data[req_col] = 'New York'
                                    elif req_col in ['Rainfall', 'Temperature']:
                                        historical_data[req_col] = 100 if req_col == 'Rainfall' else 25
                                    else:
                                        # Financial columns
                                        historical_data[req_col] = 100000
                        
                        # Display the mapped data
                        st.subheader("Mapped Data")
                        st.dataframe(historical_data)
                        
                        # Define analyze_trends function
                        def analyze_trends(df, col):
                            from sklearn.linear_model import LinearRegression
                            y = df[col].values
                            X = np.arange(len(y)).reshape(-1, 1)
                            model = LinearRegression()
                            model.fit(X, y)
                            avg_change = model.coef_[0]
                            percent_change = ((y[-1] - y[0]) / y[0]) * 100 if y[0] != 0 else 0
                            direction = "upward" if avg_change > 0 else "downward" if avg_change < 0 else "flat"
                            return {
                                "model": model,
                                "avg_change": avg_change,
                                "percent_change": percent_change,
                                "direction": direction
                            }

                        # Analyze trends in numeric columns
                        numeric_cols = historical_data.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            st.write("### Detected Trends")
                            for col in numeric_cols:
                                trend = analyze_trends(historical_data, col)
                                if trend:
                                    st.write(f"**{col}**: {trend['direction']} trend, "
                                            f"average change of {trend['avg_change']:.2f} per period, "
                                            f"total change of {trend['percent_change']:.2f}%")
                                    
                                    # Plot trend
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    ax.scatter(range(len(historical_data)), historical_data[col], color='blue')
                                    
                                    # Plot regression line
                                    X = np.array(range(len(historical_data))).reshape(-1, 1)
                                    ax.plot(X, trend['model'].predict(X), color='red', linewidth=2)
                                    
                                    ax.set_title(f"Trend Analysis: {col}")
                                    ax.set_xlabel("Time Period")
                                    ax.set_ylabel(col)
                                    ax.grid(True, linestyle='--', alpha=0.7)
                                    
                                    # Add future prediction (next 3 periods)
                                    future_X = np.array(range(len(historical_data), len(historical_data) + 3)).reshape(-1, 1)
                                    future_y = trend['model'].predict(future_X)
                                    
                                    ax.plot(future_X, future_y, color='green', linestyle='--', linewidth=2)
                                    ax.scatter(future_X, future_y, color='green', marker='x', s=100)
                                    
                                    # Add shaded area for prediction uncertainty
                                    ax.fill_between(
                                        future_X.flatten(), 
                                        future_y - trend['model'].coef_[0] * 0.2, 
                                        future_y + trend['model'].coef_[0] * 0.2, 
                                        color='green', alpha=0.2
                                    )
                                    
                                    st.pyplot(fig)
                    else:
                        # If no tables found, try to parse text as CSV
                        try:
                            historical_data = pd.read_csv(StringIO(text))
                            st.write("Extracted data from text:")
                            st.dataframe(historical_data.head())
                        except:
                            # If parsing fails, create a dataframe from text
                            historical_data = pd.DataFrame({'Data': text.split('\n')})
                            st.write("Could not extract structured data. Created text dataframe:")
                            st.dataframe(historical_data.head())
                elif file_extension == '.txt':
                    # Try different delimiters
                    for delimiter in [',', '\t', '|', ';']:
                        try:
                            historical_data = pd.read_csv(uploaded_file, delimiter=delimiter)
                            if len(historical_data.columns) > 1:
                                break
                        except:
                            pass
                    else:
                        # If no delimiter worked, read as plain text
                        uploaded_file.seek(0)
                        text = uploaded_file.read().decode('utf-8')
                        historical_data = pd.DataFrame({'Data': text.split('\n')})
                elif file_extension == '.docx':
                    try:
                        import docx
                        doc = docx.Document(uploaded_file)
                        text = "\n".join([para.text for para in doc.paragraphs])
                        # Try to parse as CSV
                        try:
                            historical_data = pd.read_csv(StringIO(text))
                        except:
                            # If parsing fails, create a dataframe from text
                            historical_data = pd.DataFrame({'Data': text.split('\n')})
                    except ImportError:
                        st.error("Please install python-docx: pip install python-docx")
                elif file_extension == '.xml':
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(uploaded_file)
                        root = tree.getroot()
                        # Try to convert XML to dataframe
                        data = []
                        for child in root:
                            row = {}
                            for elem in child:
                                row[elem.tag] = elem.text
                            data.append(row)
                        historical_data = pd.DataFrame(data)
                    except:
                        st.error("Could not parse XML file")
                elif file_extension == '.sas7bdat':
                    try:
                        historical_data = pd.read_sas(uploaded_file)
                    except:
                        st.error("Could not parse SAS file")
                elif file_extension == '.parquet':
                    try:
                        historical_data = pd.read_parquet(uploaded_file)
                    except:
                        st.error("Could not parse Parquet file")
                elif file_extension == '.feather':
                    try:
                        historical_data = pd.read_feather(uploaded_file)
                    except:
                        st.error("Could not parse Feather file")
                
                # Make the success message more visible
                st.markdown(f"""
                    <div style="background-color: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin: 15px 0; border: 1px solid #c3e6cb;">
                        <strong>Success!</strong> File {uploaded_file.name} uploaded successfully!
                    </div>
                """, unsafe_allow_html=True)
                
                # Display the data
                st.dataframe(historical_data)
                
                # Add robust error handling for the column mapping interface
                try:
                    # Column mapping interface
                    st.markdown('<h3 style="color:white;">Map Columns</h3>', unsafe_allow_html=True)
                    st.write("Please map your data columns to the required fields:")
                    
                    required_columns = ['FY', 'R&D Spend', 'Administration', 'Marketing Spend', 
                                       'Profit', 'Rainfall', 'Temperature', 'Country', 'State']
                    
                    column_mapping = {}
                    
                    # Ensure all column names are strings
                    if 'historical_data' in locals() and historical_data is not None:
                        historical_data.columns = [str(col) for col in historical_data.columns]
                        available_columns = ['None'] + list(historical_data.columns)
                        
                        for req_col in required_columns:
                            # Try to find a close match
                            default_idx = 0
                            for i, col in enumerate(available_columns):
                                # Convert to string and compare
                                col_str = str(col).lower().replace(' ', '')
                                req_col_str = req_col.lower().replace(' ', '')
                                if col_str == req_col_str:
                                    default_idx = i
                                    break
                            
                            column_mapping[req_col] = st.selectbox(
                                f"Map '{req_col}' to:", 
                                available_columns,
                                index=default_idx,
                                key=f"map_{req_col}"
                            )
                        
                        # Apply mapping button
                        if st.button("Apply Column Mapping"):
                            # Create a new dataframe with mapped columns
                            mapped_data = pd.DataFrame(index=historical_data.index)
                            
                            for req_col, mapped_col in column_mapping.items():
                                if mapped_col != 'None':
                                    mapped_data[req_col] = historical_data[mapped_col]
                                else:
                                    # Use defaults for missing columns
                                    if req_col == 'FY':
                                        # Generate fiscal years
                                        current_year = datetime.now().year
                                        mapped_data[req_col] = [f"FY{current_year-len(historical_data)+i}" for i in range(len(historical_data))]
                                    elif req_col == 'Country':
                                        mapped_data[req_col] = 'United States'
                                    elif req_col == 'State':
                                        mapped_data[req_col] = 'New York'
                                    elif req_col in ['Rainfall', 'Temperature']:
                                        mapped_data[req_col] = 100 if req_col == 'Rainfall' else 25
                                    else:
                                        # Financial columns
                                        mapped_data[req_col] = 100000
                            
                            # Update historical_data with mapped data
                            historical_data = mapped_data
                            st.success("Column mapping applied successfully!")
                            st.dataframe(historical_data)
                    else:
                        st.warning("No data available for mapping. Please upload a file first.")
                except Exception as e:
                    st.error(f"Error in column mapping: {str(e)}")
                    st.info("Using default values for all required columns.")
                    
                    # Create default data if mapping fails
                    if 'historical_data' in locals() and historical_data is not None:
                        index_length = len(historical_data)
                    else:
                        index_length = 5  # Default length
                        
                    # Create a default dataframe
                    historical_data = pd.DataFrame(index=range(index_length))
                    current_year = datetime.now().year
                    
                    # Add default columns
                    historical_data['FY'] = [f"FY{current_year-index_length+i}" for i in range(index_length)]
                    historical_data['R&D Spend'] = 100000
                    historical_data['Administration'] = 100000
                    historical_data['Marketing Spend'] = 100000
                    historical_data['Profit'] = 150000
                    historical_data['Rainfall'] = 100
                    historical_data['Temperature'] = 25
                    historical_data['Country'] = 'United States'
                    historical_data['State'] = 'New York'
                
                # Explicitly ask for historical year range instead of extracting from data
                st.markdown('<h3 style="color:white;">Specify Historical Data Range</h3>', unsafe_allow_html=True)
                
                # Create a container for the year inputs with custom styling - making labels BLACK
                st.markdown("""
                    <style>
                    /* Style for the year input container */
                    .year-inputs-container {
                        display: flex;
                        flex-direction: row;
                        justify-content: space-between;
                        align-items: flex-start;
                        margin-bottom: 15px;
                    }
                    
                    /* Style for each year input group */
                    .year-input-group {
                        width: 48%;
                    }
                    
                    /* Style for year labels - MAKING THEM BLACK */
                    .year-label {
                        color: black;
                        font-weight: bold;
                        margin-bottom: 5px;
                        display: block;
                        background-color: white;
                        padding: 5px;
                        border-radius: 4px;
                    }
                    </style>
                    
                    <div class="year-inputs-container">
                        <div class="year-input-group">
                            <span class="year-label">Start Year</span>
                        </div>
                        <div class="year-input-group">
                            <span class="year-label">End Year</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add specific styling for the year number inputs
                st.markdown("""
                    <style>
                    /* Target specifically the Start Year and End Year number inputs */
                    [data-testid="stNumberInput"] input {
                        background-color: white !important;
                        color: black !important;
                        font-weight: bold !important;
                        border: 2px solid #110361 !important;
                    }
                    
                    /* Style the plus/minus buttons */
                    [data-testid="stNumberInput"] button {
                        background-color: white !important;
                        color: #110361 !important;
                        border: 1px solid #110361 !important;
                    }
                    
                    /* Style the plus/minus button icons */
                    [data-testid="stNumberInput"] button svg {
                        fill: #110361 !important;
                    }
                    
                    /* Style the container */
                    [data-testid="stNumberInput"] {
                        background-color: white !important;
                        border-radius: 8px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    start_year = st.number_input('Start Year', 
                                               min_value=current_year-50, 
                                               max_value=current_year+50, 
                                               value=st.session_state.start_year,
                                               key='start_year_upload',
                                               label_visibility="collapsed")  # Hide the default label
                with col2:
                    end_year = st.number_input('End Year', 
                                             min_value=start_year,
                                             max_value=current_year+50, 
                                             value=max(start_year, st.session_state.end_year),
                                             key='end_year_upload',
                                             label_visibility="collapsed")  # Hide the default label
                
                # Update session state
                st.session_state.start_year = start_year
                st.session_state.end_year = end_year
                
                # Country and state selection
                st.markdown('<h3 style="color:white;">Location</h3>', unsafe_allow_html=True)
                country = st.selectbox('Country', list(COUNTRIES.keys()), key='country_upload')
                state = st.selectbox('State', COUNTRIES[country], key='state_upload')
                
                # Add this specific CSS for the forecast years input right before you create the forecast_years input
                st.markdown("""
                    <style>
                    /* Target specifically the forecast years input */
                    div[data-testid="stNumberInput"] > div[data-baseweb="input"] > input[aria-label*="Number of Years to Predict"] {
                        color: #000000 !important;
                        font-weight: bold !important;
                        background-color: #ffffff !important;
                        font-size: 16px !important;
                        border: 2px solid #110361 !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<h3 style="color:white;">Forecast Configuration</h3>', unsafe_allow_html=True)
                forecast_years = st.number_input(
                    'Number of Years to Forecast',
                    min_value=1,
                    max_value=20,
                    value=st.session_state.forecast_years,
                    key='forecast_years_upload'
                )
                
                # Create forecast range based on user input
                forecast_range = list(range(end_year + 1, end_year + 1 + forecast_years))
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
            # Close the file upload section div
            st.markdown('</div>', unsafe_allow_html=True)
        # Remove the else clause completely - don't add any alternative content

# Main content area
def create_ibus_chart(df, metric, title, show_historical=False, historical_df=None):
    fig = go.Figure()
    
    if show_historical and historical_df is not None and not historical_df.empty:
        # Create a complete list of all years (historical + forecast) with no gaps
        all_years = []
        
        # Add historical years
        if isinstance(historical_df.index[0], str) and historical_df.index[0].startswith('FY'):
            # If years are in FY format
            hist_years = [int(year[2:]) for year in historical_df.index]
            all_years.extend([f'FY{year}' for year in hist_years])
        else:
            all_years.extend(historical_df.index.tolist())
        
        # Add forecast years
        if isinstance(df.index[0], str) and df.index[0].startswith('FY'):
            # If years are in FY format
            forecast_years = [int(year[2:]) for year in df.index]
            
            # Find the gap between historical and forecast
            if hist_years[-1] + 1 < forecast_years[0]:
                # Fill the gap
                for year in range(hist_years[-1] + 1, forecast_years[0]):
                    all_years.append(f'FY{year}')
            
            all_years.extend([f'FY{year}' for year in forecast_years])
        else:
            all_years.extend(df.index.tolist())
        
        # Remove duplicates and sort
        all_years = sorted(list(set(all_years)))
        
        # Create a complete dataframe with all years
        complete_df = pd.DataFrame(index=all_years)
        
        # Add historical data
        for year in historical_df.index:
            if year in complete_df.index:
                complete_df.loc[year, metric] = historical_df.loc[year, metric]
        
        # Add forecast data
        for year in df.index:
            if year in complete_df.index:
                complete_df.loc[year, metric] = df.loc[year, metric]
        
        # Fill any gaps to ensure continuous visualization
        complete_df[metric] = complete_df[metric].interpolate(method='linear')
        
        # Determine the last historical year
        last_hist_year = historical_df.index[-1] if not historical_df.empty else None
        
        # For line charts, use a single continuous line
        if st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=complete_df.index,
                y=complete_df[metric],
                mode='lines+markers',
                line=dict(color=IBUS_PRIMARY, width=3, shape='linear'),  # Ensure linear connections
                marker=dict(
                    size=8,
                    color=[IBUS_SECONDARY if x <= last_hist_year else IBUS_ACCENT for x in complete_df.index]
                ),
                name=metric,
                connectgaps=True  # Connect any remaining gaps
            ))
            
            # Add a legend for historical vs forecast
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=IBUS_SECONDARY),
                name='Historical'
            ))
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=8, color=IBUS_ACCENT),
                name='Forecast'
            ))
        
        # Bar chart with historical and forecast data - ensure no gaps
        elif st.session_state.chart_type == 'Bar':
            # Create separate traces for historical and forecast with explicit width
            hist_years = [year for year in complete_df.index if year <= last_hist_year] if last_hist_year else []
            forecast_years = [year for year in complete_df.index if year > last_hist_year] if last_hist_year else complete_df.index
            
            # Add historical bars with width parameter
            if hist_years:
                fig.add_trace(go.Bar(
                    x=hist_years,
                    y=complete_df.loc[hist_years, metric],
                    name='Historical',
                    marker_color=IBUS_SECONDARY,
                    width=0.7  # Set width to less than 1 to create spacing
                ))
            
            # Add forecast bars with width parameter
            if forecast_years:
                fig.add_trace(go.Bar(
                    x=forecast_years,
                    y=complete_df.loc[forecast_years, metric],
                    name='Forecast',
                    marker_color=IBUS_ACCENT,
                    width=0.7  # Set width to less than 1 to create spacing
                ))
        
        # Area chart with historical and forecast data - ensure no gaps
        elif st.session_state.chart_type == 'Area':
            # Create a single continuous dataset for the area chart
            # This is the key change - we'll create one continuous series instead of two separate ones
            
            # First, determine which points are historical vs forecast
            is_historical = [x <= last_hist_year if last_hist_year else False for x in complete_df.index]
            
            # Create a continuous x-axis with no gaps
            all_x = complete_df.index.tolist()
            all_y = complete_df[metric].tolist()
            
            # Create a list of colors for each point
            colors = [IBUS_SECONDARY if h else IBUS_ACCENT for h in is_historical]
            
            # Add a single area trace with custom fill colors
            for i in range(len(all_x) - 1):
                # Add segment by segment to control the fill color
                fig.add_trace(go.Scatter(
                    x=[all_x[i], all_x[i+1]],
                    y=[all_y[i], all_y[i+1]],
                    fill='tozeroy',
                    mode='lines',
                    line=dict(color=colors[i], width=3),
                    showlegend=i == 0 or (i > 0 and is_historical[i] != is_historical[i-1]),
                    name='Historical Data' if is_historical[i] else 'Forecast Data',
                    connectgaps=True
                ))
            
            # Add legend entries
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=IBUS_SECONDARY, width=3),
                name='Historical Data',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=IBUS_ACCENT, width=3),
                name='Forecast Data',
                showlegend=True
            ))
        
        # Pie chart
        elif st.session_state.chart_type == 'Pie':
            # Create a colorscale with distinct colors for each year
            num_years = len(complete_df.index)
            colorscale = px.colors.qualitative.Plotly[:num_years] if num_years <= 10 else px.colors.qualitative.Plotly + px.colors.qualitative.D3[:num_years-10]
            
            # Create the pie chart
            fig = go.Figure(go.Pie(
                labels=complete_df.index,
                values=complete_df[metric],
                marker_colors=colorscale,
                hole=.3,
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial'
            ))
            
            # Update layout specifically for pie chart
            fig.update_layout(
                title=title,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    bgcolor="white",
                    bordercolor="lightgray",
                    borderwidth=1,
                    font=dict(color="black", size=12)
                ),
                # Remove grid lines and axes for pie chart
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(visible=False, showgrid=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=IBUS_DARK, size=12),
                height=500
            )
    else:
        # If no historical data, just show forecast
        if st.session_state.chart_type == 'Line':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines+markers',
                name='Forecast',
                line=dict(color=IBUS_PRIMARY, width=3),
                marker=dict(size=8, color=IBUS_ACCENT)
            ))
        elif st.session_state.chart_type == 'Bar':
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[metric],
                name='Forecast',
                marker_color=IBUS_PRIMARY
            ))
        elif st.session_state.chart_type == 'Area':
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines',
                fill='tozeroy',
                name='Forecast Data',
                line=dict(color=IBUS_PRIMARY, width=3)
            ))
            
            # Add a clearer legend on the right side
            fig.update_layout(
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                    bgcolor="white",
                    bordercolor="lightgray",
                    borderwidth=1,
                    font=dict(color="black", size=12)
                )
            )
        elif st.session_state.chart_type == 'Pie':
            # Create a colorscale with distinct colors for each year
            num_years = len(df.index)
            colorscale = px.colors.qualitative.Plotly[:num_years] if num_years <= 10 else px.colors.qualitative.Plotly + px.colors.qualitative.D3[:num_years-10]
            
            # Create the pie chart
            fig = go.Figure(go.Pie(
                labels=df.index,
                values=df[metric],
                marker_colors=colorscale,
                hole=.3,
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial'
            ))
            
            # Update layout specifically for pie chart
            fig.update_layout(
                title=title,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    bgcolor="white",
                    bordercolor="lightgray",
                    borderwidth=1,
                    font=dict(color="black", size=12)
                ),
                # Remove grid lines and axes for pie chart
                xaxis=dict(visible=False, showgrid=False),
                yaxis=dict(visible=False, showgrid=False),
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(color=IBUS_DARK, size=12),
                height=500
            )
    
    # Apply layout for non-pie charts
    if st.session_state.chart_type != 'Pie':
        fig.update_layout(
            title=title,
            xaxis_title="Fiscal Year",
            yaxis_title=metric,
            hovermode="x unified",
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=IBUS_DARK, size=12),
            height=500,
            bargap=0,  # Remove gap between bars
            bargroupgap=0,  # Remove gap between bar groups
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="white",
                bordercolor="lightgray",
                borderwidth=1,
                font=dict(color="black", size=12)
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=0.2,
                gridcolor='#fafafa',
                tickfont=dict(size=12, color=IBUS_DARK),
                type='category',  # Use category type to ensure no gaps
                categoryorder='array',  # Ensure order is maintained
                categoryarray=complete_df.index if 'complete_df' in locals() else df.index  # Use the complete index
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=0.2,
                gridcolor='#fafafa',
                tickfont=dict(size=12, color=IBUS_DARK)
            )
        )
    
    return fig

# Forecasting logic
if 'forecast_range' in locals() and forecast_range:
    st.subheader(f"Forecast for FY{forecast_range[0]} to FY{forecast_range[-1]}")
    
    if input_method == 'Slider Input':
        # Prepare data for forecasting
        future_preds = []
        lag_rainfall = last_known_rainfall.copy()
        lag_temp = last_known_temp.copy()
        
        # Create dummy variables for country and state
        country_dummies = {f"Country_{c}": 1 if c == country else 0 for c in COUNTRIES.keys()}
        state_dummies = {f"State_{s}": 1 if s == state else 0 for s in COUNTRIES[country]}
        
        for i, year in enumerate(forecast_range):
            # Calculate expected climate values
            rain_val = base_rainfall + (i * rainfall_trend)
            temp_val = base_temp + (i * temp_trend)
            
            # Create input data for prediction using user-defined growth rates
            new_data = pd.DataFrame({
                'R&D Spend': [rnd_spend * (1 + rnd_growth/100) ** (i+1)],  # User-defined growth rate
                'Administration': [admin_spend * (1 + admin_growth/100) ** (i+1)],  # User-defined growth rate
                'Marketing Spend': [marketing_spend * (1 + marketing_growth/100) ** (i+1)],  # User-defined growth rate
                **country_dummies,
                **state_dummies
            })

            # Add lagged climate features
            for lag in range(1, n_lags + 1):
                if i == 0:
                    # For first forecast year, use historical lags
                    new_data[f'Rainfall_lag_{lag}'] = lag_rainfall[-lag]
                    new_data[f'Temperature_lag_{lag}'] = lag_temp[-lag]
                else:
                    # For subsequent years, use previous forecast values
                    new_data[f'Rainfall_lag_{lag}'] = future_preds[i-lag][5] if i >= lag else lag_rainfall[-(lag-i)]
                    new_data[f'Temperature_lag_{lag}'] = future_preds[i-lag][6] if i >= lag else lag_temp[-(lag-i)]

            # Transform features
            new_poly = poly.transform(new_data[['R&D Spend', 'Administration', 'Marketing Spend']])
            new_poly_df = pd.DataFrame(new_poly, columns=poly_feature_names)

            # Combine all features
            lag_and_cat = new_data.drop(columns=['R&D Spend', 'Administration', 'Marketing Spend'])
            new_features = pd.concat([new_poly_df, lag_and_cat], axis=1)
            
            # Ensure all expected columns are present
            for col in feature_cols:
                if col not in new_features.columns:
                    new_features[col] = 0
            
            new_features = new_features[feature_cols]

            # Scale features and make prediction
            new_scaled = scaler.transform(new_features)
            new_scaled_df = pd.DataFrame(new_scaled, columns=feature_cols)
            
                       
            # Get model prediction
            pred = best_model.predict(new_scaled_df)[0]
            
            # Store results
            future_preds.append([f'FY{year}', 
                               rnd_spend * (1 + rnd_growth/100) ** (i+1), 
                               admin_spend * (1 + admin_growth/100) ** (i+1), 
                               marketing_spend * (1 + marketing_growth/100) ** (i+1), 
                               pred[0], rain_val, temp_val])

            # Update lagged values
            lag_rainfall.append(rain_val)
            lag_temp.append(temp_val)
        
        forecast_df = pd.DataFrame(future_preds, columns=['FY', 'R&D Spend', 'Administration', 
                                                         'Marketing Spend', 'Profit', 
                                                         'Rainfall', 'Temperature'])
    
    elif input_method == 'Manual Entry' and manual_data:
        # Prepare data for forecasting from manual entry
        manual_df = pd.DataFrame(manual_data, columns=['FY', 'R&D Spend', 'Administration', 
                                                     'Marketing Spend', 'Profit', 
                                                     'Rainfall', 'Temperature'])
        
        # Create dummy variables for country and state
        country_dummies = {f"Country_{c}": 1 if c == country else 0 for c in COUNTRIES.keys()}
        state_dummies = {f"State_{s}": 1 if s == state else 0 for s in COUNTRIES[country]}
        
        if len(manual_df) >= n_lags:
            last_known_rainfall = list(manual_df['Rainfall'].iloc[-n_lags:])
            last_known_temp = list(manual_df['Temperature'].iloc[-n_lags:])
            
            future_preds = []
            lag_rainfall = last_known_rainfall.copy()
            lag_temp = last_known_temp.copy()
            
            for i, year in enumerate(forecast_range):
                # Calculate expected climate values (using last known trend)
                rain_trend = manual_df['Rainfall'].diff().mean() if len(manual_df) > 1 else 2.0
                temp_trend = manual_df['Temperature'].diff().mean() if len(manual_df) > 1 else 0.5
                
                rain_val = manual_df['Rainfall'].iloc[-1] + (i+1)*rain_trend
                temp_val = manual_df['Temperature'].iloc[-1] + (i+1)*temp_trend
                
                # Create input data for prediction
                new_data = pd.DataFrame({
                    'R&D Spend': [manual_df['R&D Spend'].iloc[-1] * (1.02 ** (i+1))],
                    'Administration': [manual_df['Administration'].iloc[-1] * (1.01 ** (i+1))],
                    'Marketing Spend': [manual_df['Marketing Spend'].iloc[-1] * (1.03 ** (i+1))],
                    **country_dummies,
                    **state_dummies
                })

                # Add lagged climate features
                for lag in range(1, n_lags + 1):
                    if i == 0:
                        new_data[f'Rainfall_lag_{lag}'] = lag_rainfall[-lag]
                        new_data[f'Temperature_lag_{lag}'] = lag_temp[-lag]
                    else:
                        new_data[f'Rainfall_lag_{lag}'] = future_preds[i-lag][5] if i >= lag else lag_rainfall[-(lag-i)]
                        new_data[f'Temperature_lag_{lag}'] = future_preds[i-lag][6] if i >= lag else lag_temp[-(lag-i)]

                # Transform features
                new_poly = poly.transform(new_data[['R&D Spend', 'Administration', 'Marketing Spend']])
                new_poly_df = pd.DataFrame(new_poly, columns=poly_feature_names)

                # Combine all features
                lag_and_cat = new_data.drop(columns=['R&D Spend', 'Administration', 'Marketing Spend'])
                new_features = pd.concat([new_poly_df, lag_and_cat], axis=1)
                
                # Ensure all expected columns are present
                for col in feature_cols:
                    if col not in new_features.columns:
                        new_features[col] = 0
                
                new_features = new_features[feature_cols]

                # Scale features and make prediction
                new_scaled = scaler.transform(new_features)
                new_scaled_df = pd.DataFrame(new_scaled, columns=feature_cols)
                
                # Get model prediction
                pred = best_model.predict(new_scaled_df)[0]
                
                # Store results
                future_preds.append([f'FY{year}', 
                                manual_df['R&D Spend'].iloc[-1] * (1.02 ** (i+1)),
                                manual_df['Administration'].iloc[-1] * (1.01 ** (i+1)),
                                manual_df['Marketing Spend'].iloc[-1] * (1.03 ** (i+1)),
                                pred[0], rain_val, temp_val])

                # Update lagged values
                lag_rainfall.append(rain_val)
                lag_temp.append(temp_val)
            
            forecast_df = pd.DataFrame(future_preds, columns=['FY', 'R&D Spend', 'Administration', 
                                                            'Marketing Spend', 'Profit', 
                                                            'Rainfall', 'Temperature'])
        else:
            st.warning("Not enough historical data to make predictions. Need at least 3 years of data.")
            forecast_df = manual_df
    
    elif input_method == 'File Upload' and 'historical_data' in locals():
        # Process uploaded data
        if len(historical_data) >= n_lags:
            # Try to extract required columns
            required_columns = ['R&D Spend', 'Administration', 'Marketing Spend', 
                               'Profit', 'Rainfall', 'Temperature']
            
            # Check if columns exist or create defaults
            for col in required_columns:
                if col not in historical_data.columns:
                    if col == 'Profit':
                        historical_data[col] = historical_data.get('Revenue', 150000)
                    elif col in ['R&D Spend', 'Administration', 'Marketing Spend']:
                        historical_data[col] = historical_data.get(col, 100000)
                    elif col == 'Rainfall':
                        historical_data[col] = historical_data.get(col, 100)
                    elif col == 'Temperature':
                        historical_data[col] = historical_data.get(col, 25)
            
            # Default to United States/New York if country/state not specified
            if 'Country' not in historical_data.columns:
                historical_data['Country'] = 'United States'
            if 'State' not in historical_data.columns:
                historical_data['State'] = 'New York'
            
            country = historical_data['Country'].iloc[-1]
            state = historical_data['State'].iloc[-1]
            
            # Create dummy variables for country and state
            country_dummies = {f"Country_{c}": 1 if c == country else 0 for c in COUNTRIES.keys()}
            state_dummies = {f"State_{s}": 1 if s == state else 0 for s in COUNTRIES[country]}
            
            last_known_rainfall = list(historical_data['Rainfall'].iloc[-n_lags:])
            last_known_temp = list(historical_data['Temperature'].iloc[-n_lags:])
            
            future_preds = []
            lag_rainfall = last_known_rainfall.copy()
            lag_temp = last_known_temp.copy()
            
            for i, year in enumerate(forecast_range):
                # Calculate expected climate values (using last known trend)
                rain_trend = historical_data['Rainfall'].diff().mean() if len(historical_data) > 1 else 2.0
                temp_trend = historical_data['Temperature'].diff().mean() if len(historical_data) > 1 else 0.5
                
                rain_val = historical_data['Rainfall'].iloc[-1] + (i+1)*rain_trend
                temp_val = historical_data['Temperature'].iloc[-1] + (i+1)*temp_trend
                
                # Create input data for prediction
                new_data = pd.DataFrame({
                    'R&D Spend': [historical_data['R&D Spend'].iloc[-1] * (1.02 ** (i+1))],
                    'Administration': [historical_data['Administration'].iloc[-1] * (1.01 ** (i+1))],
                    'Marketing Spend': [historical_data['Marketing Spend'].iloc[-1] * (1.03 ** (i+1))],
                    **country_dummies,
                    **state_dummies
                })

                # Add lagged climate features
                for lag in range(1, n_lags + 1):
                    if i == 0:
                        new_data[f'Rainfall_lag_{lag}'] = lag_rainfall[-lag]
                        new_data[f'Temperature_lag_{lag}'] = lag_temp[-lag]
                    else:
                        new_data[f'Rainfall_lag_{lag}'] = future_preds[i-lag][5] if i >= lag else lag_rainfall[-(lag-i)]
                        new_data[f'Temperature_lag_{lag}'] = future_preds[i-lag][6] if i >= lag else lag_temp[-(lag-i)]

                # Transform features
                new_poly = poly.transform(new_data[['R&D Spend', 'Administration', 'Marketing Spend']])
                new_poly_df = pd.DataFrame(new_poly, columns=poly_feature_names)

                # Combine all features
                lag_and_cat = new_data.drop(columns=['R&D Spend', 'Administration', 'Marketing Spend'])
                new_features = pd.concat([new_poly_df, lag_and_cat], axis=1)
                
                # Ensure all expected columns are present
                for col in feature_cols:
                    if col not in new_features.columns:
                        new_features[col] = 0
                
                new_features = new_features[feature_cols]

                # Scale features and make prediction
                new_scaled = scaler.transform(new_features)
                new_scaled_df = pd.DataFrame(new_scaled, columns=feature_cols)
                
                # Get model prediction
                pred = best_model.predict(new_scaled_df)[0]
                
                # Store results
                future_preds.append([f'FY{year}', 
                                historical_data['R&D Spend'].iloc[-1] * (1.02 ** (i+1)),
                                historical_data['Administration'].iloc[-1] * (1.01 ** (i+1)),
                                historical_data['Marketing Spend'].iloc[-1] * (1.03 ** (i+1)),
                                pred[0], rain_val, temp_val])

                # Update lagged values
                lag_rainfall.append(rain_val)
                lag_temp.append(temp_val)
            
            forecast_df = pd.DataFrame(future_preds, columns=['FY', 'R&D Spend', 'Administration', 
                                                            'Marketing Spend', 'Profit', 
                                                            'Rainfall', 'Temperature'])
        else:
            st.warning("Not enough historical data to make predictions. Need at least 3 years of data.")
            forecast_df = historical_data

    # Display forecast results
    if 'forecast_df' in locals() and not forecast_df.empty:
        # Chart type selector with black text
        st.markdown("""
            <style>
            [data-testid="stSelectbox"] > label {
                color: black !important;
                font-weight: 600 !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.session_state.chart_type = st.selectbox(
            'Select Chart Type',
            ['Line', 'Bar', 'Area', 'Pie'],
            index=0,
            key='chart_type_selector'
        )
        
        # Create tabs with more explicit styling
        st.markdown("""
            <style>
            /* Force tab visibility */
            .stTabs [data-baseweb="tab-list"] {
                display: flex !important;
                flex-direction: row !important;
                gap: 10px !important;
                background-color: white !important;
                padding: 10px !important;
                border-radius: 8px !important;
                margin-bottom: 20px !important;
                border: 1px solid #e6e6e6 !important;
                overflow-x: auto !important;
            }
            .stTabs [data-baseweb="tab"] {
                display: flex !important;
                flex-direction: column !important;
                justify-content: center !important;
                align-items: center !important;
                height: auto !important;
                min-width: 120px !important;
                white-space: pre-wrap !important;
                background-color: #f8f9fa !important;
                border-radius: 4px !important;
                padding: 10px 16px !important;
                margin: 0 !important;
                font-weight: 600 !important;
                color: #0056b3 !important;
                border: 1px solid #dee2e6 !important;
                cursor: pointer !important;
            }
            .stTabs [aria-selected="true"] {
                background-color: #0056b3 !important;
                color: white !important;
                border: 1px solid #0056b3 !important;
            }
            /* Make tab content visible */
            .stTabs [data-baseweb="tab-panel"] {
                display: block !important;
                visibility: visible !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Create tabs
        tab_labels = ["Financial Forecast", "Climate Forecast", "Combined View", "Data Table"]
        tabs = st.tabs(tab_labels)
        
        with tabs[0]:
            # Financial metrics
            st.markdown(f'<h3 style="color:{IBUS_PRIMARY}; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px; margin-bottom: 16px;">Profit Forecast</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Profit</div>
                        <div class="metric-value">${forecast_df['Profit'].mean():,.0f}</div>
                        <div class="metric-change {'positive' if forecast_df['Profit'].iloc[-1] > forecast_df['Profit'].iloc[0] else 'negative'}">
                            {'↑' if forecast_df['Profit'].iloc[-1] > forecast_df['Profit'].iloc[0] else '↓'} 
                            ${abs(forecast_df['Profit'].iloc[-1] - forecast_df['Profit'].iloc[0]):,.0f} change
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Marketing ROI</div>
                        <div class="metric-value">{forecast_df['Profit'].mean() / forecast_df['Marketing Spend'].mean():,.2f}x</div>
                        <div class="metric-change positive">
                            Based on average marketing spend of ${forecast_df['Marketing Spend'].mean():,.0f}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Profit chart
            fig_profit = create_ibus_chart(
                forecast_df.set_index('FY'), 
                'Profit', 
                'Profit Forecast (FY{} - FY{})'.format(forecast_df['FY'].iloc[0], forecast_df['FY'].iloc[-1]),
                show_historical=True,
                historical_df=historical_data.set_index('FY') if 'historical_data' in locals() else None
            )
            fig_profit.update_layout(
                xaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                )
            )
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Profit Forecast Analysis</h4>', unsafe_allow_html=True)
            st.plotly_chart(fig_profit, use_container_width=True, key="tab1_profit_chart")
            
            # Add financial metrics charts
            col1, col2 = st.columns(2)
            
            with col1:
                # R&D Spend chart
                fig_rnd = create_ibus_chart(
                    forecast_df.set_index('FY'), 
                    'R&D Spend', 
                    'R&D Spend Forecast',
                    show_historical=True,
                    historical_df=historical_data.set_index('FY') if 'historical_data' in locals() and not historical_data.empty else None
                )
                fig_rnd.update_layout(
                    height=400,
                    xaxis=dict(
                        title_font=dict(size=14, color="black"),
                        tickfont=dict(size=12, color=IBUS_DARK),
                        showgrid=True, gridwidth=1, gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title_font=dict(size=14, color="black"),
                        tickfont=dict(size=12, color=IBUS_DARK),
                        showgrid=True, gridwidth=1, gridcolor='lightgray'
                    )
                )
                st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">R&D Spend Analysis</h4>', unsafe_allow_html=True)
                st.plotly_chart(fig_rnd, use_container_width=True, key="tab1_rnd_chart")
            
            with col2:
                # Marketing Spend chart
                fig_marketing = create_ibus_chart(
                    forecast_df.set_index('FY'), 
                    'Marketing Spend', 
                    'Marketing Spend Forecast',
                    show_historical=True,
                    historical_df=historical_data.set_index('FY') if 'historical_data' in locals() and not historical_data.empty else None
                )
                fig_marketing.update_layout(
                    height=400,
                    xaxis=dict(
                        title_font=dict(size=14, color="black"),
                        tickfont=dict(size=12, color=IBUS_DARK),
                        showgrid=True, gridwidth=1, gridcolor='lightgray'
                    ),
                    yaxis=dict(
                        title_font=dict(size=14, color="black"),
                        tickfont=dict(size=12, color=IBUS_DARK),
                        showgrid=True, gridwidth=1, gridcolor='lightgray'
                    )
                )
                st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Marketing Spend Analysis</h4>', unsafe_allow_html=True)
                st.plotly_chart(fig_marketing, use_container_width=True, key="tab1_marketing_chart")
            
            # Add custom styling for the historical data input fields
            st.markdown("""
                <style>
                /* Style for the historical data input container */
                .stExpander {
                    background-color: #110361 !important;
                    border-radius: 8px !important;
                    margin-bottom: 10px !important;
                    border: none !important;
                }
                
                /* Style for the expander header */
                .st-emotion-cache-1gulkj5, .st-emotion-cache-1wmy9hl {
                    background-color: #110361 !important;
                    color: white !important;
                    font-weight: bold !important;
                    font-size: 1.1rem !important;
                }
                
                /* Style for input fields */
                .stNumberInput input {
                    background-color: white !important;
                    color: #110361 !important;
                    font-weight: bold !important;
                    border: 2px solid white !important;
                    border-radius: 4px !important;
                    padding: 8px !important;
                }
                
                /* Style for input field labels */
                .stNumberInput label {
                    color: white !important;
                    font-weight: bold !important;
                }
                </style>
            """, unsafe_allow_html=True)

            # Administration Spend chart
            fig_admin = create_ibus_chart(
                forecast_df.set_index('FY'), 
                'Administration', 
                'Administration Spend Forecast (FY{} - FY{})'.format(forecast_df['FY'].iloc[0], forecast_df['FY'].iloc[-1]),
                show_historical=True,
                historical_df=historical_data.set_index('FY') if 'historical_data' in locals() else None
            )
            fig_admin.update_layout(
                height=400,
                xaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                )
            )
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Administration Spend Analysis</h4>', unsafe_allow_html=True)
            st.plotly_chart(fig_admin, use_container_width=True, key="tab1_admin_chart")
        
        with tabs[1]:
            # Climate metrics
            st.markdown(f'<h3 style="color:{IBUS_PRIMARY}; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px; margin-bottom: 16px;">Climate Forecast</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Rainfall</div>
                        <div class="metric-value">{forecast_df['Rainfall'].mean():.1f} mm</div>
                        <div class="metric-change {'positive' if forecast_df['Rainfall'].iloc[-1] > forecast_df['Rainfall'].iloc[0] else 'negative'}">
                            {'↑' if forecast_df['Rainfall'].iloc[-1] > forecast_df['Rainfall'].iloc[0] else '↓'} 
                            {abs(forecast_df['Rainfall'].iloc[-1] - forecast_df['Rainfall'].iloc[0]):.1f} mm change
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Average Temperature</div>
                        <div class="metric-value">{forecast_df['Temperature'].mean():.1f} °C</div>
                        <div class="metric-change {'positive' if forecast_df['Temperature'].iloc[-1] > forecast_df['Temperature'].iloc[0] else 'negative'}">
                            {'↑' if forecast_df['Temperature'].iloc[-1] > forecast_df['Temperature'].iloc[0] else '↓'} 
                            {abs(forecast_df['Temperature'].iloc[-1] - forecast_df['Temperature'].iloc[0]):.1f} °C change
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Climate charts
            fig_rain = create_ibus_chart(
                forecast_df.set_index('FY'), 
                'Rainfall', 
                'Rainfall Forecast (FY{} - FY{})'.format(forecast_df['FY'].iloc[0], forecast_df['FY'].iloc[-1]),
                show_historical=True,
                historical_df=historical_data.set_index('FY') if 'historical_data' in locals() else None
            )
            fig_rain.update_layout(
                xaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                )
            )
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Rainfall Forecast Analysis</h4>', unsafe_allow_html=True)
            st.plotly_chart(fig_rain, use_container_width=True, key="tab2_rain_chart")
            
            fig_temp = create_ibus_chart(
                forecast_df.set_index('FY'), 
                'Temperature', 
                'Temperature Forecast (FY{} - FY{})'.format(forecast_df['FY'].iloc[0], forecast_df['FY'].iloc[-1]),
                show_historical=True,
                historical_df=historical_data.set_index('FY') if 'historical_data' in locals() else None
            )
            fig_temp.update_layout(
                xaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, gridwidth=1, gridcolor='lightgray'
                )
            )
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Temperature Forecast Analysis</h4>', unsafe_allow_html=True)
            st.plotly_chart(fig_temp, use_container_width=True, key="tab2_temp_chart")
            
            # Add correlation between temperature and rainfall
            fig_temp_rain = px.scatter(
                forecast_df,
                x='Temperature',
                y='Rainfall',
                trendline="lowess",
                title=f"Temperature vs Rainfall Correlation (FY{forecast_df['FY'].iloc[0]} - FY{forecast_df['FY'].iloc[-1]})",
                color='FY',
                color_continuous_scale=[IBUS_PRIMARY, IBUS_SECONDARY]
            )

            # Update layout with white background
            fig_temp_rain.update_layout(
                xaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title_font=dict(size=14, color="black"),
                    tickfont=dict(size=12, color=IBUS_DARK),
                    showgrid=True, 
                    gridwidth=1, 
                    gridcolor='lightgray'
                ),
                plot_bgcolor='white',  # Set plot background to white
                paper_bgcolor='white',  # Set paper background to white
                font=dict(color='black'),  # Set font color to black
                height=400
            )
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Climate Correlation Analysis</h4>', unsafe_allow_html=True)
            st.plotly_chart(fig_temp_rain, use_container_width=True, key="tab2_temp_rain_chart")

            # Make grid lines more visible on white background
            fig_temp_rain.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig_temp_rain.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        with tabs[2]:
            # 3D visualization
            st.markdown(f'<h3 style="color:{IBUS_PRIMARY}; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px; margin-bottom: 16px;">3D Relationship Analysis</h3>', unsafe_allow_html=True)
            st.markdown(f'<h4 style="color:{IBUS_PRIMARY}; margin-top: 20px;">Financial & Climate Integrated View</h4>', unsafe_allow_html=True)
            fig_3d = px.scatter_3d(
                forecast_df,
                x='R&D Spend',
                y='Marketing Spend',
                z='Profit',
                color='Temperature',
                size='Rainfall',
                size_max=25,  # Increased maximum size
                opacity=0.9,  # Increased opacity
                title=f"Profit vs Spend vs Climate (FY{forecast_df['FY'].iloc[0]} - FY{forecast_df['FY'].iloc[-1]})",
                hover_name='FY',
                color_continuous_scale='Viridis'  # Changed to Viridis for better visibility
            )

            # Add custom color bar settings
            fig_3d.update_coloraxes(
                colorbar=dict(
                    thickness=20,
                    len=0.7,
                    title=dict(text="Temperature (°C)", font=dict(size=14, color="black")),
                    tickfont=dict(size=12, color="black"),
                    outlinewidth=1,
                    outlinecolor="black"
                )
            )

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='R&D Spend',
                    yaxis_title='Marketing Spend',
                    zaxis_title='Profit',
                    xaxis=dict(
                        showgrid=True,
                        gridwidth=2,
                        gridcolor='darkgray',  # Changed from black to darkgray
                        tickfont=dict(size=10, color='black'),
                        title_font=dict(size=12, color="black")
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridwidth=2,
                        gridcolor='darkgray',  # Changed from black to darkgray
                        tickfont=dict(size=10, color='black'),
                        title_font=dict(size=12, color="black")
                    ),
                    zaxis=dict(
                        showgrid=True,
                        gridwidth=2,
                        gridcolor='darkgray',  # Changed from black to darkgray
                        tickfont=dict(size=10, color='black'),
                        title_font=dict(size=12, color="black")
                    ),
                    bgcolor='rgba(240,240,240,0.8)'  # Light gray background for better contrast
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                font=dict(color='black'),
                paper_bgcolor='white',
                plot_bgcolor='white'
            )

            # Update traces for better visibility
            fig_3d.update_traces(
                marker=dict(
                    line=dict(width=1, color='black'),  # Add black outline to markers
                )
            )
            st.plotly_chart(fig_3d, use_container_width=True, key="tab3_3d_chart")
        
        with tabs[3]:
            # Data table view
            st.markdown(f'<h3 style="color:{IBUS_PRIMARY}; border-bottom: 2px solid {IBUS_PRIMARY}; padding-bottom: 8px; margin-bottom: 16px;">Forecast Data</h3>', unsafe_allow_html=True)
            
            # Define colors
            very_light_blue = "#e6f2ff"
            medium_dark_blue = "#2c3e50"  # Changed from black to a medium dark blue
            
            # Apply styling with background color and text color
            styled_df = forecast_df.style.format({
                'R&D Spend': '${:,.0f}',
                'Administration': '${:,.0f}',
                'Marketing Spend': '${:,.0f}',
                'Profit': '${:,.0f}',
                'Rainfall': '{:.1f} mm',
                'Temperature': '{:.1f} °C'
            }).set_properties(**{
                'background-color': very_light_blue,
                'color': medium_dark_blue,
                'font-weight': '500'
            })
            
            # Add custom CSS to change the header and index colors
            st.markdown(f"""
                <style>
                    /* Change header and index background color */
                    .dataframe th {{
                        background-color: {IBUS_SECONDARY} !important;
                        color: white !important;
                        font-weight: bold !important;
                    }}
                    
                    /* Change index column styling */
                    .dataframe tbody tr th {{
                        background-color: {IBUS_SECONDARY} !important;
                        color: white !important;
                        font-weight: bold !important;
                    }}
                    
                    /* Add border styling */
                    .dataframe td, .dataframe th {{
                        border: 1px solid #ddd !important;
                    }}
                </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                styled_df,
                use_container_width=True
            )
            
            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Forecast Data as CSV",
                data=csv,
                file_name=f"ibus_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )

# Add this CSS specifically for the FY expander headings in Manual Entry
st.markdown("""
    <style>
    /* Target expander headers specifically in Manual Entry */
    div[data-testid="stExpander"] > div:first-child {
        background-color: #110361 !important;
    }
    
    /* Make the expander header text black for better visibility */
    div[data-testid="stExpander"] > div:first-child p,
    div[data-testid="stExpander"] > div:first-child span,
    div[data-testid="stExpander"] > div:first-child button {
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    /* Style the expander content background */
    div[data-testid="stExpander"] > div:nth-child(2) {
        background-color: #110361 !important;
    }
    
    /* Make input labels inside expanders black */
    div[data-testid="stExpander"] [data-testid="stNumberInput"] label {
        color: black !important;
        background-color: white !important;
        font-weight: bold !important;
        padding: 2px 5px !important;
        border-radius: 4px !important;
        margin-bottom: 5px !important;
        display: inline-block !important;
    }
    
    /* Style the input fields */
    div[data-testid="stExpander"] [data-testid="stNumberInput"] input {
        background-color: white !important;
        color: black !important;
        font-weight: bold !important;
        border: 2px solid white !important;
    }
    </style>
""", unsafe_allow_html=True)

