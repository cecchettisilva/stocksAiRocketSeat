
#Importação de libs que serão usadas:
import json 
import os 
from datetime import datetime

import yfinance as yf
from crewai import Agent, Task, Crew, Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

#Criando o Yahoo Finance Tool
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stock prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func= lambda ticket: fetch_stock_price(ticket)
)

#Importando OpenAI LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-4o-mini")


#AGENTE 1 - Analíse preço histórico
stockPriceAnalyst = Agent(
    role = "Senior stock price analyst",
    goal = "find the {ticket} stock price and analyses trends",
    backstory = """You're a highly experienced in analyzing the price of an specifed stock
    and make predictions about its future price.
    """,
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory = True,
    allow_delegation = False,
    tools= [yahoo_finance_tool]
) 

#TAREFA do AGENTE 1
getStockPrice = Task(
    description = "Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output = """Specify the current trend stock price - up, down or sideaways. 
    eg. stock = 'AAPL', price UP
    """,
    agent = stockPriceAnalyst
)

#TOOL AGENT 2 - importando tool de search 
search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

#AGENTE 2 - Analíse de noticias da empresa
newsAnalyst = Agent(
    role = "Stock news analyst",
    goal = """create a short summary of the market news related to the stock {ticket} company. Specify the current trend - up, down and sideways
    with the news context. For each request stock asset, specify a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
    """,
    backstory = """You're a highly experienced in analyzing the market trends and news and have have tracked assets for more 10 years.
    You're also master level analyst in the tradicional markets and have the understanding of human psychology.
    You understand news, theirs titles and information, but you look at those with a health dose of skeptism.
    You consider also the source of the news articles.
    """,
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory = True,
    allow_delegation = False,
    tools= [search_tool]
)

#TAREFA do AGENTE 2 
#current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

get_news = Task(
    description = f"""Take the stock and always include BTC to it (if not request)
    Use the search tool to search each one individually.
    The current date {datetime.now()}.
    Compose the results into a helpfull report.
    """,
    expected_output = """A summary of the overeall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news.
    Use the format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICITION>
    <FEAR/GREED SCORE>
    """,
    agent = newsAnalyst
)

#AGENTE 3 - Avaliação da ação
stockAnalystWrite = Agent(
    role = "Senior stock analyst Writer",
    goal = "Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend",
    backstory = """You're widely accepted as the best stock analyst in the market.
    You understand complex concepts and create compelling stories and narratives that resonate with wider audiences.
    You understand macro factors and combine multiple theories.
    eg: cycle theory and fundamental analyses.
    You're able to hold multiple opnions when analyzing anything.
    """,
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory = True,
    allow_delegation = True,
    tools= []
)

#TAREFA do AGENTE 3
writeAnalyses = Task(
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company,
    that is brief and highlights the most import points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trends and news summay.
    """,
    expected_output = """An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner.
    It should contain:
    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest 
    - Main part provides the meat of the anaysis including the news summary and fear/greed scores
    - Summary - key facts and concrete future trend prediciton - up, down or sideways
    """,
    agent = stockAnalystWrite,
    context = [getStockPrice, get_news]
)

#Criando grupos de agente de IA
crew = Crew(
    agents= [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks= [getStockPrice, get_news, writeAnalyses],
    verbose=True,
    process= Process.hierarchical, 
    full_output=True,
    share_crew=False,
    manager_llm= llm,
    max_iter=15
)

#Criando grupos de agente de IA
crew = Crew(
    agents= [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks= [getStockPrice, get_news, writeAnalyses],
    verbose=True,
    process= Process.hierarchical, 
    full_output=True,
    share_crew=False,
    manager_llm= llm,
    max_iter=15
)

with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label = "Run Research")
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])