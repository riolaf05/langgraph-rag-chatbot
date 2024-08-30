import sys 
sys.path.append(r'C:\Users\ELAFACRB1\Codice\GitHub\rio-utils-app\src')
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
import sqlite3
import pandas as pd

DATABASE_NAME="climbing_gear_customers.sql"
TABLE_NAME="customer_interactions"

conn=sqlite3.connect(DATABASE_NAME)
db = SQLDatabase.from_uri("sqlite:///"+DATABASE_NAME+".db")

model = ChatGroq(
    temperature=0, 
    model_name="Llama3-8b-8192"
    )


toolkit = SQLDatabaseToolkit(db=db, llm=model) #SQLDatabaseToolkit comes with 4 pre-built tools that help the model to work with tables in a database.
