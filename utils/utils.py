# import logging
# import requests
# import json
# import time
# import boto3
# import os
# import io
# import openai
# from langchain.prompts import PromptTemplate
# from langchain.llms import OpenAI
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain, SimpleSequentialChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
# from langchain.chains.summarize import load_summarize_chain
# from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
# from langchain.document_loaders.generic import GenericLoader
# from langchain.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
# from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
# from langchain.docstore.document import Document
# from langchain.chains.question_answering import load_qa_chain
# from langchain.retrievers import ParentDocumentRetriever
# from langchain.embeddings import HuggingFaceBgeEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.document_loaders import RSSFeedLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from urllib.request import urlopen
# import spacy
# import chromadb
# import numpy as np
# import random
# import datetime
# import moviepy.editor as mp
# import speech_recognition as sr
# import re

# import whisper

# from config.environments import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION

# TRANSCRIBE_BUCKET = 's3://newsp4-transcribe-docs-bucket'

# AWS Texttract
# class AWSTexttract:

#     def __init__(self):
#         self.client = boto3.client('textract', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)

#     def get_text(self, file_path):

#         if type(file_path) == str:
#             #cioè se passo il path del file
#             with open(file_path, 'rb') as file:
#                 img_test = file.read()
#                 bytes_test = bytearray(img_test)
#                 print('Image loaded', file_path)
#             response = self.client.detect_document_text(Document={'Bytes': bytes_test})
#         else:
#             #se passo il formato PIL
#             buf = io.BytesIO()
#             file_path.save(buf, format='JPEG')
#             byte_im = buf.getvalue()
#             response = self.client.detect_document_text(Document={'Bytes': byte_im})

#         text = ''
#         for item in response["Blocks"]:
#             if item["BlockType"] == "LINE":
#                 text += item["Text"] + '\n'

#         return text

# # AWS Transcribe
# class AWSTranscribe:

#         def __init__(self, job_uri, region):
#             self.transcribe = boto3.client('transcribe', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=region)
#             self.job_verification = False
#             self.job_uri=job_uri

#         def generate_job_name(self):
#             return "stt_"+str(time.time_ns())+"_"+str(random.randint(0,500))

#         def check_job_name(self, job_name):
#             """
#             Check if the transcribe job name is existed or not
#             """
#             self.job_verification = True
#             # all the transcriptions
#             existed_jobs = self.transcribe.list_transcription_jobs()
#             for job in existed_jobs['TranscriptionJobSummaries']:
#                 if job_name == job['TranscriptionJobName']:
#                     self.job_verification = False
#                 break
#             # if job_verification == False:
#             #     command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
#             #     if command.lower() == "y" or command.lower() == "yes":
#             #         self.transcribe.delete_transcription_job(TranscriptionJobName=job_name)
#                 # elif command.lower() == "n" or command.lower() == "no":
#                 #     job_name = input("Insert new job name? ")
#                 #     self.check_job_name(job_name)
#                 # else:
#                 #     print("Input can only be (Y/N)")
#                 #     command = input(job_name + " has existed. \nDo you want to override the existed job (Y/N): ")
#             return job_name

#         def amazon_transcribe(self, job_uri, job_name, audio_file_name, language):
#             """
#             For single speaker
#             """
#             # Usually, I put like this to automate the process with the file name
#             # "s3://bucket_name" + audio_file_name
#             # Usually, file names have spaces and have the file extension like .mp3
#             # we take only a file name and delete all the space to name the job
#             job_uri = os.path.join('s3://'+job_uri, audio_file_name)
#             # file format
#             file_format = audio_file_name.split('.')[-1]

#             # check if name is taken or not
#             job_name = self.check_job_name(job_name)
#             print('Transctiption started from:')
#             print(job_uri)
#             self.transcribe.start_transcription_job(
#                 TranscriptionJobName=job_name,
#                 Media={'MediaFileUri': job_uri},
#                 MediaFormat = file_format,
#                 LanguageCode=language)

#             while True:
#                 result = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
#                 if result['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
#                     break
#                 time.sleep(15)
#             if result['TranscriptionJob']['TranscriptionJobStatus'] == "COMPLETED":
#                 response = urlopen(result['TranscriptionJob']['Transcript']['TranscriptFileUri'])
#                 data = json.loads(response.read())
#             return data['results']['transcripts'][0]['transcript']

# # AWS S3
# class AWSS3:

#         def __init__(self, bucket=None):
#             self.s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)
#             self.bucket = bucket

#         def read_metadata(self, key, id):
#             response = self.s3_client.head_object(Bucket=self.bucket, Key=key)
#             return response['Metadata'][id]

#         def list_items(self, key):
#             list=self.s3_client.list_objects_v2(Bucket=self.bucket,Prefix=key)
#             return list.get('Contents', [])

#         def upload_file(self, fileobj, key):
#             """Upload a file to an S3 bucket
#             """
#             try:
#                 #the first argument is the file path

#                 self.s3_client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs={'Metadata': {'Name': key}})
#                 logging.info('File Successfully Uploaded on S3')
#                 return True
#             except FileNotFoundError:
#                 time.sleep(9)
#                 logging.error('File not found.')
#                 return False

#         def delete_file(self, object_name):
#             """Delete a file from an S3 bucket
#             :param object_name: S3 object name
#             :return: True if file was deleted, else False
#             """
#             try:
#                 self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
#                 return True
#             except Exception as e:
#                 logging.error(e)
#                 return False

#         def download_file(self, object_name, file_name):
#             """Download a file from an S3 bucket

#             :param bucket: Bucket to download from
#             :param object_name: S3 object name
#             :param file_name: File to download, path
#             :return: True if file was downloaded, else False

#             """
#             # Download the file
#             try:
#                 response = self.s3_client.download_file(self.bucket, object_name, file_name)
#             except Exception as e:
#                 logging.error(e)
#                 return False
#             return True

#         def copy_file(self, target_bucket, target_key, dest_bucket):
#             """Copy a file from an S3 bucket
#             :param dest: S3 destination
#             :param target: S3 target
#             :return: True if file was copied, else False
#             """
#             try:
#                 copy_source = {
#                     'Bucket': target_bucket,
#                     'Key': target_key
#                     }
#                 bucket = boto3.resource('s3').Bucket(dest_bucket) #resource method https://stackoverflow.com/questions/70293628/attributeerror-s3-object-has-no-attribute-bucket
#                 bucket.copy(copy_source, target_key)
#                 return True
#             except Exception as e:
#                 logging.error(e)
#                 return False

# # Lambda
# class AWSLambda:
#     def __init__(self):
#         self.lambda_client = boto3.client('lambda', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, region_name=AWS_REGION)

#     def invoke_lambda(self, function_name, payload):
#         """Invoke a lambda function
#         """
#         try:
#             response = self.lambda_client.invoke(FunctionName=function_name, InvocationType='RequestResponse', Payload=payload)
#             logging.info('Lambda invoked')
#             body=response['Payload'].read()
#             json_object = json.loads(body)
#             return json_object['body']
#         except Exception as e:
#             logging.error(e)
#             return False

# class SpeechToText:

#     def __init__(self, model, bucket):
#         self.model = model
#         self.bucket = bucket

#     def transcribe(self, file_path):
#         '''
#         Takes a file path in ingress and returns a text in output
#         '''

#         # if self.model == 'whisper-base':
#         #     model = whisper.load_model("base")
#         #     text = model.transcribe(file_path)
#         #     return text['text']

#         if self.model == 'transcribe':
#             transcribe = AWSTranscribe(self.bucket, 'us-east-1')
#             job_name=transcribe.generate_job_name()
#             text = transcribe.amazon_transcribe(self.bucket, job_name, file_path, 'it-IT')
#             return text

#         elif self.model == 'openai':
#             # Step 1: Extract audio from video
#             audio_file = extract_audio(file_path)
#             # Step 2: Convert speech to text
#             text = speech_to_text(audio_file)
#             # Step 3: Clean the text
#             cleaned_text = clean_text(text)
#             # Step 4: Input text to OpenAI API
#             improvised_text = openai_api(cleaned_text)
#             return improvised_text

#         else:
#             pass
#             #TODO implement new models!!

#         # Function to extract audio from video file
#         def extract_audio(video_file):
#             video = mp.VideoFileClip(video_file)
#             audio = video.audio
#             audio_file = "extracted_audio.wav"
#             audio.write_audiofile(audio_file)
#             return audio_file

#         # Function to perform speech to text conversion
#         def speech_to_text(audio_file):
#             recognizer = sr.Recognizer()
#             with sr.AudioFile(audio_file) as source:
#                 audio = recognizer.record(source)
#                 text = recognizer.recognize_google(audio)
#                 return text

#         # Function to clean text
#         def clean_text(text):
#             # Remove stammering words (words repeated more than twice)
#             cleaned_text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
#             # Remove punctuation and special characters
#             cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned_text)
#             return cleaned_text


#         def openai_api(text):
#             prompt = "Be precise and rewrite the context and topic in thie video in simple words"
#             content = prompt + " " + text

#             OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#             client = OpenAI(
#                 api_key=OPENAI_API_KEY,
#             )
#             response = client.chat.completions.create(
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": content,
#                     }
#                 ],
#                 model="gpt-3.5-turbo",
#                 )
#             clean_text =  response.choices[0].message.content
#             # print(message_content)
#             return clean_text


# class TextSplitter:

#     def __init__(self,
#                  chunk_size=2000,
#                  chunk_overlap=0
#                  ):
#         # self.nlp = spacy.load("it_core_news_sm")
#         self.chunk_size=chunk_size
#         self.chunk_overlap=chunk_overlap
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", " "],
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#             )

#     # def process(self, text):
#     #     doc = self.nlp(text)
#     #     sents = list(doc.sents)
#     #     vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
#     #     return sents, vecs

#     # def cluster_text(self, sents, vecs, threshold):
#     #     clusters = [[0]]
#     #     for i in range(1, len(sents)):
#     #         if np.dot(vecs[i], vecs[i-1]) < threshold:
#     #             clusters.append([])
#     #         clusters[-1].append(i)

#     #     return clusters


#     # def clean_text(self, text):
#     #     # Add your text cleaning process here
#     #     return text

#     # def semantic_split_text(self, data, threshold=0.3):
#     #     '''
#     #     Split thext using semantic clustering and spacy see https://getpocket.com/read/3906332851
#     #     '''


#     #     # Initialize the clusters lengths list and final texts list
#     #     clusters_lens = []
#     #     final_texts = []

#     #     # Process the chunk
#     #     sents, vecs = self.process(data)

#     #     # Cluster the sentences
#     #     clusters = self.cluster_text(sents, vecs, threshold)

#     #     for cluster in clusters:
#     #         cluster_txt = self.clean_text(' '.join([sents[i].text for i in cluster]))
#     #         cluster_len = len(cluster_txt)

#     #         # Check if the cluster is too short
#     #         if cluster_len < 60:
#     #             continue

#     #         # Check if the cluster is too long
#     #         elif cluster_len > 3000:
#     #             threshold = 0.6
#     #             sents_div, vecs_div = self.process(cluster_txt)
#     #             reclusters = self.cluster_text(sents_div, vecs_div, threshold)

#     #             for subcluster in reclusters:
#     #                 div_txt = self.clean_text(' '.join([sents_div[i].text for i in subcluster]))
#     #                 div_len = len(div_txt)

#     #                 if div_len < 60 or div_len > 3000:
#     #                     continue

#     #                 clusters_lens.append(div_len)
#     #                 final_texts.append(div_txt)

#     #         else:
#     #             clusters_lens.append(cluster_len)
#     #             final_texts.append(cluster_txt)

#     #     #converting to Langchain documents
#     #     ##lo posso fare anche con .create_documents !!
#     #     # final_docs=[]
#     #     # for doc in final_texts:
#     #     #     final_docs.append(Document(page_content=doc, metadata={"source": "local"}))

#     #     return final_texts

#     def create_langchain_documents(self, texts, metadata):
#         final_docs=[]
#         if type(texts) == str:
#             texts = [texts]
#         for doc in texts:
#             final_docs.append(Document(page_content=doc, metadata=metadata))
#         return final_docs

#     #fixed split
#     def fixed_split(self, data):
#         '''
#         Takes Langchain documents as input
#         Returns splitted documents
#         '''
#         docs = self.text_splitter.split_documents(data)
#         return docs

# DynamoDB
# class DynamoDBManager:
#     def __init__(self, region, table_name):
#         self.region = region
#         self.table_name = table_name
#         self.dynamodb = boto3.resource('dynamodb', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY,  region_name=region)
#         self.table = self.dynamodb.Table(table_name)

#     def write_item(self, item):
#         try:
#             response = self.table.put_item(Item=item)
#             print("Item added successfully:", response)
#         except Exception as e:
#             print("Error writing item:", e)

#     def update_item(self, key, update_expression, expression_values):
#         try:
#             response = self.table.update_item(
#                 Key=key,
#                 UpdateExpression=update_expression,
#                 ExpressionAttributeValues=expression_values
#             )
#             print("Item updated successfully:", response)
#         except Exception as e:
#             print("Error updating item:", e)
#     def get_item(self, key):
#         try:
#             response = self.table.get_item(Key=key)
#             print("Item retrieved successfully:", response)
#             return response
#         except Exception as e:
#             print("Error retrieving item:", e)

# Embedder
# class EmbeddingFunction:
#     def __init__(self, parametro):
#         self.parametro = parametro

#         if parametro == 'openAI':
#             # self.embedder = OpenAIEmbeddings(model="text-embedding-3-large")
#             self.embedder = OpenAIEmbeddings()

#         # elif parametro == 'sentenceTransformers':
#         #     self.embedder = SentenceTransformerEmbeddings('all-MiniLM-L6-v2')

#         elif parametro == 'bgeEmbedding':
#             model_name = "BAAI/bge-small-en-v1.5"
#             encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
#             bge_embeddings = HuggingFaceBgeEmbeddings(
#                 model_name=model_name,
#                 model_kwargs={'device': 'cuda'},
#                 encode_kwargs=encode_kwargs
#             )
#             self.embedder = bge_embeddings


#         elif parametro == 'hkunlpEmbedding':
#             model_name = "hkunlp/instructor-large"
#             encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
#             bge_embeddings = HuggingFaceBgeEmbeddings(
#                 model_name=model_name,
#                 model_kwargs={'device': 'cuda'},
#                 encode_kwargs=encode_kwargs
#             )
#             self.embedder = bge_embeddings
#         # else:
#         #     self.embedder = self.default_method


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
# import qdrant_client
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
# from langchain.vectorstores import Qdrant
# from langchain.indexes import index
# from langchain.indexes import SQLRecordManager

# class QDrantDBManager:
#     def __init__(self,
#                  url,
#                  port,
#                  collection_name,
#                  vector_size, #openAI embedding,
#                  embedding,
#                  record_manager_url,
#                  ):
#         self.url=url
#         self.port=port
#         self.collection_name=collection_name
#         self.vector_size=vector_size
#         self.embedding=embedding
#         self.client = QdrantClient(url, port=6333)
#         self.record_manager_url=record_manager_url

#         try:
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=VectorParams(size=vector_size, distance=Distance.DOT)
#             )
#         except Exception as e:
#             print(f"Collection {self.collection_name} already exists!")
#             pass

#         self.vector_store = Qdrant(
#             client=self.client,
#             collection_name=self.collection_name,
#             embeddings=embedding
#         )

#         #create schema in metadata database
#         self.record_manager = SQLRecordManager(
#             f"qdranrt/{self.collection_name}",
#             db_url=record_manager_url,
#         )
#         self.record_manager.create_schema()


#     def index_documents(self, docs, cleanup="full"):
#         '''
#         Takes splitted Langchain list of documents as input
#         Write data on QDrant and hashes on local SQL DB

#         When content is mutated (e.g., the source PDF file was revised) there will be a period of time during indexing when both the new and old versions may be returned to the user.
#         This happens after the new content was written, but before the old version was deleted.

#         * incremental indexing minimizes this period of time as it is able to do clean up continuously, as it writes.
#         * full mode does the clean up after all batches have been written.
#         '''
#         index(
#             docs,
#             self.record_manager,
#             self.vector_store,
#             cleanup=cleanup,
#             source_id_key="source"
#         )

# VertexAI + Gemini
# See https://cloud.google.com/vertex-ai/docs/generative-ai/start/quickstarts/quickstart-multimodal
# import vertexai
# from vertexai.preview.generative_models import GenerativeModel, Part

# class GeminiAI:
#     def generate_text(project_id: str, location: str) -> str:
#         # Initialize Vertex AI
#         vertexai.init(project=project_id, location=location)
#         # Load the model
#         multimodal_model = GenerativeModel("gemini-pro-vision")
#         # Query the model
#         response = multimodal_model.generate_content(
#             [
#                 # Add an example image
#                 Part.from_uri(
#                     "gs://generativeai-downloads/images/scones.jpg", mime_type="image/jpeg"
#                 ),
#                 # Add an example query
#                 "what is shown in this image?",
#             ]
#         )
#         print(response)
#         return response.text

# LangChain
# class LangChainAI:

#     def __init__(self,
#                  model_name="gpt-3.5-turbo-16k",
#                  chatbot_model="gpt-3.5-turbo"
#                  ):

#         self.chatbot_model=chatbot_model
#         self.llm = ChatOpenAI(
#           model_name=model_name, # default model
#           temperature=0.9
#           ) #temperature dictates how whacky the output should be
#         self.chains = []

#     def split_docs(self, documents):
#         '''
#         Takes a list of document as an array
#         Splitting the documents into chunks of text
#         converting them into a list of documents
#         '''
#         docs = self.text_splitter.create_documents(documents)
#         # docs = text_splitter.split_documents(documents)
#         return docs

#     def translate_text(self, text):
#         prompt_template = PromptTemplate.from_template(
#             "traduci {text} in italiano."
#         )
#         prompt_template.format(text=text)
#         llmchain = LLMChain(llm=self.llm, prompt=prompt_template)
#         res=llmchain.run(text)+'\n\n'
#         return res

#     def clean_text(self, docs):
#         '''
#         Making the text more understandable by clearing unreadeable stuff,
#         using the chain StuffDocumentsChain:
#         this chain will take a list of documents,
#         inserts them all into a prompt, and passes that prompt to an LLM
#         See: https://python.langchain.com/docs/use_cases/summarization
#         '''

#         #FIXME!!

#         # Define prompt
#         prompt_template = """Rendi questo testo comprensibile mantenendo comunque il testo originale nella sua interezza:
#         "{text}"
#         Resto comprensibile:"""
#         prompt = PromptTemplate.from_template(prompt_template)

#         # Define LLM chain
#         llm_chain = LLMChain(llm=self.llm, prompt=prompt)

#         # Define StuffDocumentsChain
#         stuff_chain = StuffDocumentsChain(
#             llm_chain=llm_chain, document_variable_name="text"
#         )
#         res=stuff_chain.run(docs)
#         return res

#     def summarize_text(self, docs):
#         '''
#         Takes docs in input, produce a text in output

#         The map reduce documents chain first applies an LLM chain to each document individually (the Map step),
#         treating the chain output as a new document.
#         It then passes all the new documents to a separate combine documents chain to get a single output (the Reduce step).
#         It can optionally first compress, or collapse,
#         the mapped documents to make sure that they fit in the combine documents chain
#         (which will often pass them to an LLM). This compression step is performed recursively if necessary.
#         '''

#         #map
#         map_template = """Di seguito un testo lungo diviso in documenti:
#         {docs}
#         Basansoti su questa lista di documenti, per favore crea un riassunto per ciascuno di essi.
#         Riassunto:"""
#         map_prompt = PromptTemplate.from_template(map_template)
#         map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

#         # Reduce
#         reduce_template = """Di seguito una lista di riassunti:
#         {docs}
#         Prendi queste informazioni e sintetizzale in un riassunto finale e consolidato dei temi principali.
#         Risposta:"""
#         reduce_prompt = PromptTemplate.from_template(reduce_template)
#         reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

#         # Combines and iteratively reduces the mapped documents
#         combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
#         reduce_documents_chain = ReduceDocumentsChain(
#             # This is final chain that is called.
#             combine_documents_chain=combine_documents_chain,
#             # If documents exceed context for `StuffDocumentsChain`
#             collapse_documents_chain=combine_documents_chain,
#             # The maximum number of tokens to group documents into.
#             token_max=4000,
#         )

#         # Combining documents by mapping a chain over them, then combining results
#         map_reduce_chain = MapReduceDocumentsChain(
#             # Map chain
#             llm_chain=map_chain,
#             # Reduce chain
#             reduce_documents_chain=reduce_documents_chain,
#             # The variable name in the llm_chain to put the documents in
#             document_variable_name="docs",
#             # Return the results of the map steps in the output
#             return_intermediate_steps=False,
#         )

#         return map_reduce_chain.run(docs)

#     def bullet_point_text(self, docs):
#         '''
#         Making the text more understandable by creating bullet points,
#         using the chain StuffDocumentsChain:
#         this chain will take a list of documents,
#         inserts them all into a prompt, and passes that prompt to an LLM
#         See: https://python.langchain.com/docs/use_cases/summarization
#         '''
#         #map
#         map_template = """Di seguito un testo lungo diviso in documenti:
#         {docs}
#         Basansoti su questa lista di documenti, per favore crea un riassunto per ciascuno di essi.
#         Riassunto:"""
#         map_prompt = PromptTemplate.from_template(map_template)
#         map_chain = LLMChain(llm=self.llm, prompt=map_prompt)

#         # Reduce
#         reduce_template = """Di seguito una lista di riassunti:
#         {docs}
#         Prendi queste informazioni e sintetizzale in un elenco puntato finale che contiene i temi principali trattati..
#         Risposta:"""
#         reduce_prompt = PromptTemplate.from_template(reduce_template)
#         reduce_chain = LLMChain(llm=self.llm, prompt=reduce_prompt)

#         # Combines and iteratively reduces the mapped documents
#         combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_variable_name="docs")
#         reduce_documents_chain = ReduceDocumentsChain(
#             # This is final chain that is called.
#             combine_documents_chain=combine_documents_chain,
#             # If documents exceed context for `StuffDocumentsChain`
#             collapse_documents_chain=combine_documents_chain,
#             # The maximum number of tokens to group documents into.
#             token_max=4000,
#         )

#         # Combining documents by mapping a chain over them, then combining results
#         map_reduce_chain = MapReduceDocumentsChain(
#             # Map chain
#             llm_chain=map_chain,
#             # Reduce chain
#             reduce_documents_chain=reduce_documents_chain,
#             # The variable name in the llm_chain to put the documents in
#             document_variable_name="docs",
#             # Return the results of the map steps in the output
#             return_intermediate_steps=False,
#         )

#         return map_reduce_chain.run(docs)

#     def paraphrase_text(self, text):
#         '''
#         Paraphrasing the text using the chain
#         '''
#         prompt = PromptTemplate(
#         input_variables=["long_text"],
#         template="Puoi parafrasare questo testo (in italiano)? {long_text} \n\n",
#         )
#         llmchain = LLMChain(llm=self.llm, prompt=prompt)
#         res=llmchain.run(text)+'\n\n'
#         return res

#     def expand_text(self, text):
#         '''
#         Enhancing the text using the chain
#         '''
#         prompt = PromptTemplate(
#         input_variables=["long_text"],
#         template="Puoi arricchiere l'esposizione di questo testo (in italiano)? {long_text} \n\n",
#         )
#         llmchain = LLMChain(llm=self.llm, prompt=prompt)
#         res=llmchain.run(text)+'\n\n'
#         return res

#     def draft_text(self, text):
#         '''
#         Makes a draft of the text using the chain
#         '''
#         prompt = PromptTemplate(
#         input_variables=["long_text"],
#         template="Puoi fare una minuta della trascrizione di una riunione contenuta in questo testo (in italiano)? {long_text} \n\n",
#         )
#         llmchain = LLMChain(llm=self.llm, prompt=prompt)
#         res=llmchain.run(text)+'\n\n'
#         return res

#     def chat_prompt(self, text):
#         #TODO
#         pass

#     def extract_video(self, url):
#         '''
#         Estrae il testo di un video da un url in ingresso
#         '''
#         local = False
#         text=""
#         save_dir=""
#         # Transcribe the videos to text
#         if local:
#             loader = GenericLoader(
#                 YoutubeAudioLoader([url], save_dir), OpenAIWhisperParserLocal()
#             )
#         else:
#             loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
#         docs = loader.load()
#         for docs in docs:
#             #write all the text into the var
#             text+=docs.page_content+'\n\n'
#         return text

#     def github_prompt(self, url):
#         #TODO
#         pass

#     def summarize_repo(self, url):
#         #TODO
#         pass

#     def generate_paragraph(self, text):
#         #TODO
#         pass

#     def final_chain(self, user_questions):
#         # Generating the final answer to the user's question using all the chains

#         sentences=[]

#         for text in user_questions:
#             # print(text)

#             # Chains
#             prompt = PromptTemplate(
#                 input_variables=["long_text"],
#                 template="Puoi rendere questo testo più comprensibile? {long_text} \n\n",
#             )
#             llmchain = LLMChain(llm=self.llm, prompt=prompt)
#             res=llmchain.run(text)+'\n\n'
#             print(res)
#             sentences.append(res)

#         print(sentences)

#         # Chain 2
#         template = """Puoi ordinare il testo di queste frasi secondo il significato? {sentences}\n\n"""
#         prompt_template = PromptTemplate(input_variables=["sentences"], template=template)
#         question_chain = LLMChain(llm=self.llm, prompt=prompt_template, verbose=True)

#         # Final Chain
#         template = """Puoi sintetizzare questo testo in una lista di bullet points utili per la comprensione rapida del testo? '{text}'"""
#         prompt_template = PromptTemplate(input_variables=["text"], template=template)
#         answer_chain = LLMChain(llm=self.llm, prompt=prompt_template)

#         overall_chain = SimpleSequentialChain(
#             chains=[question_chain, answer_chain],
#             verbose=True,
#         )

#         res = overall_chain.run(sentences)

#         return res

#     def create_chatbot_chain(self):
#         model_name = self.chatbot_model
#         llm = ChatOpenAI(model_name=model_name)
#         chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
#         return chain

#     def filter_datetime_metadata(self, docs):
#         '''
#         Takes a list of documents in input
#         '''
#         for doc in docs:
#             doc.metadata['source'] = 'rss'
#             if isinstance(doc.metadata['publish_date'], datetime.datetime):
#                 # print(doc.metadata['publish_date'])
#                 doc.metadata['publish_date'] = doc.metadata['publish_date'].strftime("%Y-%m-%d")

#     def filter_newline_content(self, docs):
#         '''
#         Takes a list of documents in input
#         '''
#         for doc in docs:
#             doc.page_content = doc.page_content.replace('\n', ' ')
#             doc.metadata['source'] = 'html'
#         return docs

#     def rss_loader(self, feed):
#         splitted_docs=[]
#         urls = [feed] #TODO: change for multiple?
#         loader = RSSFeedLoader(urls=urls)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0) #FIXME
#         for doc in data:
#             splitted_docs.append(text_splitter.split_documents(data))
#         self.filter_datetime_metadata(splitted_docs[0])
#         logging.info("RSS scraping completed...scraped {} documents".format(len(splitted_docs[0])))
#         return splitted_docs[0]

#     def webpage_loader(self, url):
#         splitted_docs=[]
#         loader = WebBaseLoader(url)
#         data = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "],chunk_size=2000, chunk_overlap=0) #FIXME
#         for doc in data:
#             splitted_docs.append(text_splitter.split_documents(data))
#         self.filter_newline_content(splitted_docs[0])
#         logging.info("Web pages scraping completed...scraped {} documents".format(len(splitted_docs[0])))
#         return data


# parent document retriever
# https://github.com/azharlabs/medium/blob/main/notebooks/LangChain_RAG_Parent_Document_Retriever.ipynb?source=post_page-----5bd5c3474a8a--------------------------------
# def
