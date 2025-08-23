import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
import os

# Import the agent
from agent import agent

# Page configuration
st.set_page_config(
    page_title="Finance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No CSS - using only Streamlit native components

def load_stock_data(symbol, period="1y"):
    """Load stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if not df.empty:
            df = df.reset_index()
            df = df.rename(columns={'Date': 'Date', 'Close': 'Close', 'Volume': 'Volume'})
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if df is None or df.empty:
        return df
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def create_enhanced_price_chart(df, symbol, period="1y"):
    """Create an enhanced interactive price chart"""
    if df is None or df.empty:
        return None
    
    try:
        # Calculate technical indicators
        chart_df = calculate_technical_indicators(df)
        
        # Create subplots: price chart and volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} Stock Price Chart', 'Volume'),
            row_width=[0.7, 0.3]
        )
        
        # Main price chart
        fig.add_trace(
            go.Scatter(
                x=chart_df['Date'],
                y=chart_df['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add Bollinger Bands
        if 'BB_Upper' in chart_df.columns and 'BB_Lower' in chart_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['BB_Upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    hovertemplate='<b>%{x}</b><br>BB Upper: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['BB_Lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.1)',
                    hovertemplate='<b>%{x}</b><br>BB Lower: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add moving averages
        if 'SMA_20' in chart_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1.5),
                    hovertemplate='<b>%{x}</b><br>SMA 20: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in chart_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=chart_df['Date'],
                    y=chart_df['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='purple', width=1.5),
                    hovertemplate='<b>%{x}</b><br>SMA 50: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add volume chart
        if 'Volume' in chart_df.columns:
            colors = ['red' if close < open else 'green' for close, open in zip(chart_df['Close'], chart_df['Close'].shift(1))]
            colors[0] = 'green'  # First bar
            
            fig.add_trace(
                go.Bar(
                    x=chart_df['Date'],
                    y=chart_df['Volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7,
                    hovertemplate='<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Update layout
        period_label = {
            "1d": "1 Day",
            "5d": "5 Days", 
            "1mo": "1 Month",
            "3mo": "3 Months",
            "6mo": "6 Months",
            "1y": "1 Year",
            "2y": "2 Years",
            "5y": "5 Years",
            "10y": "10 Years",
            "ytd": "Year to Date",
            "max": "All Time"
        }.get(period, period)
        
        fig.update_layout(
            title=f'{symbol} Stock Analysis - {period_label}',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=700,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        if 'Volume' in chart_df.columns:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def display_key_metrics(df, symbol):
    """Display key metrics at the top"""
    if df is None or df.empty:
        return
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['Close'].iloc[-1]
    previous_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        st.metric("Price Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
    
    with col3:
        volume = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
        st.metric("Volume", f"{volume:,.0f}")
    
    with col4:
        period_high = df['Close'].max()
        st.metric("Period High", f"${period_high:.2f}")
    
    with col5:
        period_low = df['Close'].min()
        st.metric("Period Low", f"${period_low:.2f}")

def render_agent_sidebar():
    """Render the agent chat interface in the sidebar"""
    with st.sidebar:
        st.header('🤖 AI Financial Analyst', divider=True)
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Initialize agent memory for this session (don't clear - let it build context)
        if 'agent_memory_initialized' not in st.session_state:
            st.session_state.agent_memory_initialized = True
            # Don't clear memory - let the agent build conversation context
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                chat = st.chat_message('human')
                chat.markdown(message['content'])
            elif message['role'] == 'assistant':
                chat = st.chat_message('ai')
                chat.markdown(message['content'])
        
        # Chat input at the bottom
        user_input = st.chat_input('Ask me about stocks, analysis, or financial data...')
        
        # Handle input
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Display user message
            chat = st.chat_message('human')
            chat.markdown(user_input)
            
            # Get and display agent response using agent's memory
            chat = st.chat_message('ai')
            with st.spinner("🤖 Thinking..."):
                # Use the agent's run method which automatically handles memory
                # The agent is configured with num_history_runs=3, so it will remember 3 previous conversations
                response = agent.run(user_input)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response_text
                })
                
                chat.markdown(response_text)
        
        # Memory status (hidden from user)
        if st.session_state.get('chat_history', []):
            memory_count = len(st.session_state.chat_history) // 2  # Each conversation has user + assistant
            # Hidden memory counter for debugging only
            st.session_state.memory_count = memory_count

# Main app
def main():
    # Render agent sidebar
    render_agent_sidebar()
    
    # Initialize dashboard state
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = None
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = "MSFT"
    if 'current_period' not in st.session_state:
        st.session_state.current_period = "1y"
    
    # Input controls at the top
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        symbol = st.text_input("Enter Stock Symbol:", value=st.session_state.current_symbol).upper()
    
    with col2:
        period = st.selectbox(
            "Time Period:",
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
            index=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"].index(st.session_state.current_period)
        )
    
    with col3:
        load_button = st.button("Load Data", type="primary")
    
    # Update session state when inputs change
    if symbol != st.session_state.current_symbol:
        st.session_state.current_symbol = symbol
    if period != st.session_state.current_period:
        st.session_state.current_period = period
    
    # Page title
    st.title("📈 Stock Dashboard")
    
    # Load data and display dashboard
    if load_button or st.session_state.dashboard_data is not None:
        # Load new data only if button is clicked or no data exists
        if load_button or st.session_state.dashboard_data is None:
            with st.spinner("Loading stock data..."):
                df = load_stock_data(symbol, period)
                if df is not None:
                    st.session_state.dashboard_data = df
                else:
                    st.error(f"Could not load data for {symbol}")
                    return
        else:
            # Use existing data from session state
            df = st.session_state.dashboard_data
        
        # Display dashboard with data
        if df is not None:
            # Dashboard header
            st.markdown(f"## 📊 {symbol} Analysis Dashboard")
            
            # Display key metrics
            display_key_metrics(df, symbol)
            
            # Create tabs
            tab1, tab2, tab3 = st.tabs(["Price Chart", "Technical Indicators", "Downloads"])
            
            with tab1:
                # Create and display chart
                fig = create_enhanced_price_chart(df, symbol, period)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Failed to create chart")
            
            with tab2:
                st.subheader("Technical Indicators")
                if df is not None:
                    # Calculate indicators
                    df_indicators = calculate_technical_indicators(df)
                    
                    # Display RSI
                    if 'RSI' in df_indicators.columns:
                        st.line_chart(df_indicators.set_index('Date')['RSI'])
                        st.caption("RSI (Relative Strength Index)")
                    
                    # Display moving averages
                    if 'SMA_20' in df_indicators.columns and 'SMA_50' in df_indicators.columns:
                        ma_data = df_indicators[['Date', 'Close', 'SMA_20', 'SMA_50']].set_index('Date')
                        st.line_chart(ma_data)
                        st.caption("Moving Averages")
            
            with tab3:
                st.subheader("Download Data")
                if df is not None:
                    # Create CSV for download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{symbol}_{period}_data.csv",
                        mime="text/csv"
                    )
                    
                    # Display data table
                    st.subheader("Data Preview")
                    st.dataframe(df.tail(10))

if __name__ == "__main__":
    main()
