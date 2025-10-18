#  Finance Agent 
## Features

### **Stock Dashboard**
- **Real-time Data**: Live stock prices and market data via Yahoo Finance
- **Technical Indicators**: SMA, Bollinger Bands, RSI, and volume analysis
- **Key Metrics**: Current price, price change, volume, period highs/lows
- **Data Export**: Download stock data as CSV files

###  **AI Financial Analyst**
- **Smart Conversations**: Natural language queries about stocks and financial data
- **Memory System**: Remembers up to 3 previous conversations for context
- **Financial Expertise**: Specialized in stock analysis, ratios, and market insights
- **Real-time Data**: Uses YFinance tools for accurate, up-to-date information

##  Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/DanielCecim/Finance-Agent
   cd FinanceAgent2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

```
##  Usage

### Dashboard
1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)
2. Select a time period (1 day to 10 years)
3. Click "Load Data" to view charts and metrics
4. Explore different tabs for price charts, technical indicators, and data downloads

### AI Agent
1. Use the chat interface in the sidebar
2. Ask questions like:
   - "What is the current ratio of MSFT?"
   - "Show me AAPL's stock price"
   - "What are the key financial ratios for Tesla?"
   - "Can you remind me what is the current ratio?" (agent remembers context)

##  Configuration

### Agent Settings
The AI agent is configured with:
- **Model**: GPT-4o-mini
- **Memory**: SQLite database with 3-conversation history
- **Tools**: YFinance integration for real-time data
- **Specialization**: Financial analysis and stock market insights

### Customization
- Modify `prompts/analista.md` to change agent behavior
- Update `agent.py` to add new tools or change memory settings
- Customize charts and indicators in `streamlit_app.py`

##  Dependencies

- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **YFinance**: Yahoo Finance data integration
- **Pandas**: Data manipulation and analysis
- **Agno**: AI agent framework
- **OpenAI**: Language model integration

---
**Happy Trading! 📈**
