from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.memory.v2.memory import Memory
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.tools.yfinance import YFinanceTools
from agno.playground import Playground, serve_playground_app
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from datetime import date


from dotenv import load_dotenv
load_dotenv()

memory = Memory(
    model=OpenAIChat(id="gpt-4o-mini"),
    db=SqliteMemoryDb(table_name="user_memories",db_file="tmp/agent.db")
)

tools_list = [
    YFinanceTools(
        stock_price=True,
        company_info=True,
        stock_fundamentals=True,
        income_statements=True,
        key_financial_ratios=True,
        analyst_recommendations=True,
        company_news=True,
        technical_indicators=True,
        historical_prices=True
    ),
]

# AGENT
agent = Agent(
    name = "Finance Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=tools_list,
    memory=memory,
    enable_agentic_memory = True,
    add_history_to_messages=True,
    num_history_runs=3,
    show_tool_calls=True,
    instructions=open("prompts/analista.md", encoding='utf-8').read(),
)  

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("agent:app", reload=True)
