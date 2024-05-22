import os
import telepot
from telepot.loop import MessageLoop
import datetime
import time
from utils import langgraph
# from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pprint import pprint
load_dotenv(override=True)

llm = ChatGroq(
    temperature=0, 
    model_name="Llama3-8b-8192"
    )
graph_genereator = langgraph.Graph(langgraph.GraphState(), llm)
graph=graph_genereator.create_graph()

TELEGRAM_CHAT_ID=os.getenv('TELEGRAM_CHAT_ID')
TELEGRAM_GROUP_ID=os.getenv('TELEGRAM_GROUP_ID')

def on_chat_message(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    if content_type == 'text':
        #name = msg["first_name"]
        input_text = msg['text']
        input_text = input_text.split(" ")
        
        try:
            
            inputs = {"question": " ".join(input_text)}
            for output in graph.stream(inputs):
                for key, value in output.items():
                    pprint(f"Finished running: {key}:")
            
            # response = bot.sendPhoto(TELEGRAM_GROUP_ID, f)
            response = bot.sendMessage(TELEGRAM_GROUP_ID, value["generation"])
            
        except Exception as e:
            print(e)
            response = bot.sendMessage(TELEGRAM_GROUP_ID, "Mi dispiace non so rispondere!")

bot = telepot.Bot(TELEGRAM_CHAT_ID)
bot.message_loop(on_chat_message)
print('Ciao...')

while 1:
    time.sleep(10)