import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import sqlite3

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Fraud Detection Dashboard"

# Define the app layout
app.layout = html.Div(
    style={'backgroundColor': '#f4f7f6', 'fontFamily': 'sans-serif'},
    children=[
        html.H1(
            "Real-Time Fraud Detection Dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50', 'padding': '20px'}
        ),
        html.Div(id='live-update-text', style={'textAlign': 'center', 'paddingBottom': '20px'}),
        
        dcc.Graph(id='fraud-pie-chart'),
        dcc.Graph(id='probability-histogram'),
        
        # Interval component to trigger updates
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds (update every 5 seconds)
            n_intervals=0
        )
    ]
)

# Define the callback to update the dashboard
@app.callback(
    [Output('live-update-text', 'children'),
     Output('fraud-pie-chart', 'figure'),
     Output('probability-histogram', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    # Connect to the database and load data
    try:
        with sqlite3.connect('predictions.db') as conn:
            df = pd.read_sql_query("SELECT * FROM predictions", conn)
    except Exception as e:
        print(f"Database error: {e}")
        return html.Div('Database error'), {}, {}

    if df.empty:
        return html.Div('No prediction data yet.'), {}, {}

    # --- Calculate Stats ---
    total_predictions = len(df)
    fraud_count = df['is_fraud'].sum()
    fraud_percentage = (fraud_count / total_predictions) * 100
    
    # --- Create Figures ---
    # 1. Pie Chart for Fraud vs. Non-Fraud
    pie_fig = px.pie(
        df,
        names='is_fraud',
        title='Fraud vs. Non-Fraud Predictions',
        labels={'0': 'Not Fraud', '1': 'Fraud'},
        color_discrete_map={0: 'lightgreen', 1: 'crimson'}
    )
    
    # 2. Histogram of Fraud Probabilities
    hist_fig = px.histogram(
        df,
        x='fraud_probability',
        nbins=20,
        title='Distribution of Fraud Probabilities',
        labels={'fraud_probability': 'Fraud Probability Score'}
    )
    hist_fig.update_layout(bargap=0.1)

    # --- Update Text ---
    update_text = f"Total Predictions: {total_predictions} | Fraudulent: {fraud_count} ({fraud_percentage:.2f}%)"

    return html.Div(update_text), pie_fig, hist_fig


if __name__ == '__main__':
    app.run(debug=True, port=8050)