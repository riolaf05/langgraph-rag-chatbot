from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

llm = ChatGroq(
    temperature=0, 
    model_name="Llama3-8b-8192"
    )

topics = ["eventi, locali, ristoranti"]

### Router

# Topics should be dynamically fetched and updated whenever a new topic gets put in the vector store

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Sei un esperto nell'inoltrare una domanda 
        utente a un archivio vettoriale o a una ricerca web.  Usa l'archivio vettoriale per le domande sui seguenti argomenti: {topics}. Non è necessario essere rigidi con le parole chiave.
        Usa i tool relativi ad AWS per le richieste relative alle operazioni su strumenti cloud come accendere/spegnere VM EC2, lanciare lambda, accendere/spegnere database, etc. 
        Non usare tool AWS per domande teoriche relative agli strumenti Cloud (per le quali è necessario usare una ricerca web). 
        Altrimenti, per tutto il resto, utilizza la ricerca web.  
        Fornisci una scelta binaria 'web_search', 'aws_tool' o 'vectorstore' in base alla domanda. 
        Restituisci un JSON con una singola chiave 'datasource' e senza preamboli o spiegazioni.
        Domanda da inoltrare: Domande da indirizzare: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["topics", "question"],
)
partial_prompt = prompt.partial(topics=', '.join(topics))
question_router = partial_prompt | llm | JsonOutputParser()

### Generate
# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Sei un assistente per compiti di domande e risposte.
    Usa i seguenti pezzi di contesto recuperati per rispondere alla domanda. Se non conosci la risposta, dì semplicemente che non la conosci.
    Usa un massimo di tre frasi e mantieni la risposta concisa. 
    Fornisci la risposta con una lista nella quale ogni elemento rispetta questa struttura:
    1. Nome luogo/evento
    2. Descrizione
    3. Indirizzo se presente
    4. Sito web/email/numero di telefono se presente
    Restituisci la risposta in markdown.
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    Domanda: {question}
    Contesto: {context}
    Risposta: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "document"],
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()

### Retrieval Grader

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Sei un valutatore che valuta la rilevanza 
    di un documento recuperato rispetto a una domanda dell'utente. 
    Se il documento contiene parole chiave relative alla domanda dell'utente, valutalo come rilevante. 
    Non deve essere un test rigoroso. L'obiettivo è filtrare i recuperi errati. 
    Dai un punteggio binario 'sì' o 'no' per indicare se il documento è rilevante per la domanda. 
    Fornisci il punteggio binario come un JSON con una singola chiave 'score' e senza preamboli o spiegazioni.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm | JsonOutputParser()

### Hallucination Grader

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> Sei un valutatore che valuta se 
    una risposta è basata su / supportata da un insieme di fatti. Dai un punteggio binario 'sì' o 'no' per indicare 
    se la risposta è basata su / supportata da un insieme di fatti. Fornisci il punteggio binario come un JSON con una 
    singola chiave 'score' e senza preamboli o spiegazioni. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Ecco i fatti:
    \n ------- \n
    {documents}
    \n ------- \n
    Ecco la risposta: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)
hallucination_grader = prompt | llm | JsonOutputParser()

### Answer Grader
# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Sei un valutatore che valuta se 
    una risposta è utile per risolvere una domanda. Dai un punteggio binario 'sì' o 'no' per indicare se la risposta è 
    utile per risolvere una domanda. Fornisci il punteggio binario come un JSON con una singola chiave 'score' e senza preamboli o spiegazioni.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Di seguito la domanda: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)
answer_grader = prompt | llm | JsonOutputParser()

