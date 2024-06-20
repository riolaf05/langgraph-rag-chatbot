from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph

from utils.nodes import web_search, retrieve, grade_documents, generate
from utils.edges import route_question, decide_to_generate, grade_generation_v_documents_and_question

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]

# Graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

# Edges
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")

workflow.set_conditional_entry_point( #setta il router iniziale
    route_question,
    {
        "websearch": "websearch",
        "vectorstore": "retrieve"
    },
)

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)


workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)


app = workflow.compile()
