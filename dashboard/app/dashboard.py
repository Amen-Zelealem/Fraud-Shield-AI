import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import requests
import pandas as pd

# Initialize Dash app
app = dash.Dash(__name__)

# Define Flask backend URL
FLASK_BACKEND_URL = "http://127.0.0.1:5000"  # Flask is running on port 5000

# Function to fetch data from Flask API
def fetch_data(endpoint):
    try:
        response = requests.get(f"{FLASK_BACKEND_URL}{endpoint}")
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {endpoint}: {e}")
        return None

# Layout of the Dash app
app.layout = html.Div([
    # Title with emoji
    html.H1("📊 Fraud Insights Dashboard 🔍", style={
        'textAlign': 'center', 
        'color': '#2c3e50', 
        'fontFamily': 'Arial, sans-serif',
        'marginBottom': '20px'
    }),

    # Summary boxes
    html.Div([
        html.Div([
            html.H3("💰 Total Transactions", style={'color': '#34495e'}),
            html.Div(id='total-transactions', style={
                'fontSize': '24px', 
                'fontWeight': 'bold',
                'color': '#27ae60'
            })
        ], style={
            'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
            'border': '2px solid #ecf0f1', 'borderRadius': '10px',
            'padding': '10px', 'margin': '10px', 'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)'
        }),
        html.Div([
            html.H3("⚠️ Total Fraud Cases", style={'color': '#34495e'}),
            html.Div(id='total-fraud-cases', style={
                'fontSize': '24px', 
                'fontWeight': 'bold',
                'color': '#e74c3c'
            })
        ], style={
            'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
            'border': '2px solid #ecf0f1', 'borderRadius': '10px',
            'padding': '10px', 'margin': '10px', 'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)'
        }),
        html.Div([
            html.H3("📊 Fraud Percentage", style={'color': '#34495e'}),
            html.Div(id='fraud-percentage', style={
                'fontSize': '24px', 
                'fontWeight': 'bold',
                'color': '#8e44ad'
            })
        ], style={
            'width': '30%', 'display': 'inline-block', 'textAlign': 'center',
            'border': '2px solid #ecf0f1', 'borderRadius': '10px',
            'padding': '10px', 'margin': '10px', 'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)'
        }),
    ], style={'textAlign': 'center', 'padding': '20px'}),

    # Fraud Trends Line Chart
    html.H2("📈 Fraud Trends Over Time", style={'color': '#2c3e50', 'textAlign': 'center'}),
    dcc.Graph(id='fraud-trends-chart', style={'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),

    # Fraud by Location Map
    html.H2("🗺️ Fraud by Location", style={'color': '#2c3e50', 'textAlign': 'center'}),
    dcc.Graph(id='fraud-location-map', style={'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)', 'margin': '20px'}),

    # Fraud by Device and Browser Bar Chart
    html.H2("💻 Fraud by Device and Browser", style={'color': '#2c3e50', 'textAlign': 'center'}),
    dcc.Graph(id='fraud-device-browser-chart', style={'boxShadow': '5px 5px 10px rgba(0, 0, 0, 0.1)', 'margin': '20px'})
], style={
    'backgroundColor': '#f9f9f9', 
    'padding': '20px', 
    'fontFamily': 'Arial, sans-serif'
})

# Callback to update summary boxes
@app.callback(
    [Output('total-transactions', 'children'),
     Output('total-fraud-cases', 'children'),
     Output('fraud-percentage', 'children')],
    [Input('total-transactions', 'id')] # Just need an initial trigger, so using the ID itself.
)
def update_summary(dummy_input):
    summary_data = fetch_data('/api/fraud_summary')
    if summary_data:
        return (
            f"🌟 {summary_data['total_transactions']:,}",
            f"🚨 {summary_data['total_fraud_cases']:,}",
            f"{summary_data['fraud_percentage']:.2f}%"
        )
    else:
        return "Error", "Error", "Error"

# Callback to update fraud trends chart
@app.callback(
    Output('fraud-trends-chart', 'figure'),
    [Input('fraud-trends-chart', 'id')]
)
def update_fraud_trends_chart(dummy_input):
    trends_data = fetch_data('/api/fraud_trends')
    if trends_data:
        df = pd.DataFrame(trends_data)
        fig = px.line(df, x='purchase_date', y='fraud_cases', title='Fraud Cases Over Time', 
                      color_discrete_sequence=['#e74c3c'])
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9')
        return fig
    else:
        return {}

# Callback to update fraud location map
@app.callback(
    Output('fraud-location-map', 'figure'),
    [Input('fraud-location-map', 'id')]
)
def update_fraud_location_map(dummy_input):
    location_data = fetch_data('/api/fraud_by_location')
    if location_data:
        df = pd.DataFrame(location_data)
        fig = px.choropleth(
            df,
            locations="country",
            locationmode="country names",
            color="fraud_cases",
            hover_name="country",
            color_continuous_scale=px.colors.sequential.Plasma,
            title="Fraud Cases by Country"
        )
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9')
        return fig
    else:
        return {}

# Callback to update fraud device/browser chart
@app.callback(
    Output('fraud-device-browser-chart', 'figure'),
    [Input('fraud-device-browser-chart', 'id')]
)
def update_fraud_device_browser_chart(dummy_input):
    device_browser_data = fetch_data('/api/fraud_by_device_browser')
    if device_browser_data:
        df = pd.DataFrame(device_browser_data)
        fig = px.bar(df, x=['device_id', 'browser'], y='fraud_cases', title='Fraud Cases by Device and Browser',
                     color_discrete_sequence=['#3498db'])
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f2')
        return fig
    else:
        return {}

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)