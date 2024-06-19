from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv(override=True)

web_search_tool = TavilySearchResults(k=5)