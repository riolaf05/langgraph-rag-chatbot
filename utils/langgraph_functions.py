# from utils import langgraph
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import Dict, TypedDict, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union
from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation
from langchain.schema import HumanMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import operator
import functools
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import json 

from tools.tavily import web_search_tool as tavily_tool
from tools.retrieval_eventi import get_relevant_document_tool as retrieval_tool
from tools.aws import ec2_turnon_tools, ec2_shutdown_tools

tools = [
    tavily_tool,
    retrieval_tool,
    ec2_turnon_tools,
    ec2_shutdown_tools,
]
tool_executor = ToolExecutor(tools)

# This defines the object that is passed between each node
# in the graph. We will create different nodes for each agent and tool
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

def tool_node(state):
    """This runs tools in the graph
    It takes in an agent action and calls that tool and returns the result."""
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    tool_input = json.loads(
        last_message.additional_kwargs["function_call"]["arguments"]
    )
    # We can pass single-arg inputs by value
    if len(tool_input) == 1 and "__arg1" in tool_input:
        tool_input = next(iter(tool_input.values()))
    tool_name = last_message.additional_kwargs["function_call"]["name"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=tool_input,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(
        content=f"{tool_name} response: {str(response)}", name=action.tool
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}

def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [convert_to_openai_function(t) for t in tools]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                " Sei un assistente AI utile, collaborando con altri assistenti."
                " Utilizza gli strumenti forniti per progredire verso la risposta alla domanda."
                " Se non sei in grado di rispondere completamente, va bene, un altro assistente con strumenti diversi "
                " aiuterà dove ti sei fermato. Esegui ciò che puoi per fare progressi."
                " Se tu o qualsiasi altro assistente avete la risposta finale o il risultato,"
                " prefissa la tua risposta con FINAL ANSWER in modo che il team sappia di fermarsi."
                " Hai accesso ai seguenti strumenti: {tool_names}.\\n{system_message}"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)

def agent_node(state, agent, name):
    '''
    This helper function to create a node for a given agent
    '''
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name,
    }

def create_node_from_agent(agent, name):
    return functools.partial(agent_node, agent=agent, name=name)

# Either agent can decide to end
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if "function_call" in last_message.additional_kwargs:
        # The previus agent is invoking a tool
        return "call_tool"
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return "end"
    return "continue"