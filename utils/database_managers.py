# DynamoDB
import boto3
from langchain.docstore.document import Document
import os

class DynamoDBManager:
    def __init__(self, region, table_name):
        self.region = region
        self.table_name = table_name
        self.dynamodb = boto3.resource(
            "dynamodb",
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region,
        )
        self.table = self.dynamodb.Table(table_name)

    def write_item(self, item):
        try:
            response = self.table.put_item(Item=item)
            print("Item added successfully:", response)
        except Exception as e:
            print("Error writing item:", e)

    def update_item(self, key, update_expression, expression_values):
        try:
            response = self.table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values,
            )
            print("Item updated successfully:", response)
        except Exception as e:
            print("Error updating item:", e)

    def get_item(self, key):
        try:
            response = self.table.get_item(Key=key)
            print("Item retrieved successfully:", response)
            return response
        except Exception as e:
            print("Error retrieving item:", e)


# Chroma vector DB
# class ChromaDBManager:

#     def __init__(self):
#         self.client = chromadb.PersistentClient(path=os.getenv("PERSIST_DIR_PATH"))

#     def get_or_create_collection(self, collection_name):
#         try:
#             collection = self.client.get_or_create_collection(name=collection_name)
#             print(f"Collection {collection_name} created successfully.")
#         except Exception as e:
#             print(f"Error creating collection {collection_name}: {e}")
#         return collection

#     def store_documents(self, collection, docs):
#         '''
#         Stores document to a collection
#         Gets Langchain documents in input.
#         By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings.
#         '''
#         #add documents to collection
#         collection_documents = [document.page_content for document in docs]
#         collection_metadata = [document.metadata for document in docs]

#         #get a str id for each collection id, starting from the current maximum id of the collection
#         collection_ids = [str(collection_id + 1) for collection_id in range(len(collection_documents))]

#         #filter metadata
#         self.replace_empty_medatada(collection_metadata)

#         #add documents to collection
#         #this method creates the embedding and the colection
#         #By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings.
#         collection.add(ids=collection_ids, documents=collection_documents, metadatas=collection_metadata)

#         #the collection are automatically stored since we're using a persistant client
#         return collection.count()


#     def replace_empty_medatada(self, metadata_list):
#         #iter through metadata elements
#         for metadata in metadata_list:
#             #get index of metadata element
#             index = metadata_list.index(metadata)
#             #get metadata keys
#             metadata_keys = metadata.keys()
#             #iter through metadata keys
#             for key in metadata_keys:
#                 #if key is empty
#                 if metadata[key] == []:
#                     #replace it with None
#                     metadata_list[index][key] = ''
#                 if type(metadata[key]) == datetime.datetime:
#                     #replace it str
#                     metadata_list[index][key] = str(metadata[key])

#     def retrieve_documents(collection, query, n_results=3):
#         '''
#         To run a similarity search,
#         you can use the query method of the collection.
#         '''
#         llm_documents = []

#         #similarity search <- #TODO compare with Kendra ?
#         res=collection.query(query_texts=[query], n_results=n_results)

#         #create documents from collection
#         documents=[document for document in res['documents'][0]]
#         metadatas=[metadata for metadata in res['metadatas'][0]]

#         for i in range(len(documents)):
#             doc=Document(page_content=documents[i], metadata=metadatas[i])
#             llm_documents.append(doc)
#         return llm_documents


# Qdrant
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain.indexes import index
from langchain.indexes import SQLRecordManager


class QDrantDBManager:
    def __init__(
        self,
        url: str,
        port: int,
        collection_name: str,
        vector_size: int,  # openAI embedding,
        embedding,
        record_manager_url: str,
    ):
        self.url = url
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.embedding = embedding
        self.client = QdrantClient(url, port=6333, api_key=os.getenv('QDRANT_API_KEY'))
        self.record_manager_url = record_manager_url

        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
            )
        except Exception as e:
            print(f"Collection {self.collection_name} already exists!")
            pass

        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=embedding,
        )

        # create schema in metadata database
        self.record_manager = SQLRecordManager(
            f"qdranrt/{self.collection_name}",
            db_url=record_manager_url,
        )
        self.record_manager.create_schema()

    def index_documents(self, docs: list[dict], cleanup: str = "full"):
        """
        Takes splitted Langchain list of documents as input
        Write data on QDrant and hashes on local SQL DB

        When content is mutated (e.g., the source PDF file was revised) there will be a period of time during indexing when both the new and old versions may be returned to the user.
        This happens after the new content was written, but before the old version was deleted.

        * incremental indexing minimizes this period of time as it is able to do clean up continuously, as it writes.
        * full mode does the clean up after all batches have been written.
        """
        doc_batch = [
            Document(
                page_content=doc["source"],
                metadata={"source": doc["source"], "embedding": doc["embedding"]},
            )
            for doc in docs
        ]
        index(
            doc_batch,
            self.record_manager,
            self.vector_store,
            cleanup=cleanup,
            source_id_key="source",
        )
