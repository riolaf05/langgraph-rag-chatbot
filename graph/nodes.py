from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from graph.state import State
import datetime
from langchain_groq import ChatGroq
from tools.sql import toolkit
from tools.rag import trekking_tool

tools = [trekking_tool, toolkit.get_tools()[0], toolkit.get_tools()[1], toolkit.get_tools()[2], toolkit.get_tools()[3]]

model = ChatGroq(
    temperature=0, 
    model_name="Llama3-8b-8192"
    )


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            name = configuration.get("name", None)
            state = {**state, "user_info": name}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}




prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support for a climbing company. "
            "given customer's name, make sure to first query the db and then incorporate customer's information from the database. "
            " Use the provided tools to search for climbing gears and trekkings in Nepal, upon customer's request. "
            "\n\nCurrent user:\n<User>\n{user_info}\n</User>"

        ),
        ("placeholder", "{messages}"),
    ]
)


assistant_runnable = prompt | model.bind_tools(tools)