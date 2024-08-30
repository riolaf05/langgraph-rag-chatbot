from langchain.tools.retriever import create_retriever_tool
import sys 
sys.path.append(r'C:\Users\ELAFACRB1\Codice\GitHub\rio-utils-app\src')
sys.path.append(r'..')
from utils.embedding import EmbeddingFunction
from utils.database_managers import QDrantDBManager
import os

DATABASE_NAME="climbing_gear_customers.sql"
QDRANT_URL=os.getenv('QDRANT_URL')
COLLECTION_NAME="climbing-gear-customers"

embedding = EmbeddingFunction('fast-bgeEmbedding').embedder
qdrantClient = QDrantDBManager(
    url=QDRANT_URL,
    port=6333,
    collection_name=COLLECTION_NAME,
    vector_size=768,
    embedding=embedding,
    record_manager_url="sqlite:///record_manager_cache.sql"
)

trekking_tool = create_retriever_tool(
    qdrantClient.vector_store.as_retriever(),
    "trekking_search",
    """
    Search and return information about trekkings in Nepal.
    """
)