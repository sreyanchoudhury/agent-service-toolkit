from datetime import datetime
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from core import get_model, settings
from my_tools import vector_search_tool, youtube_search_tool


tools=[vector_search_tool, youtube_search_tool]

current_date = datetime.now().strftime("%B %d, %Y")
instructions = f"""
    You are a helpful support assistant with the ability to search the web and use other tools.
    Today's date is {current_date}.

    NOTE: THE USER CAN'T SEE THE TOOL RESPONSE.

    A few things to remember:
    - Please include markdown-formatted links to any citations used in your response. Only include one
    or two citations per response unless more are needed. ONLY USE LINKS RETURNED BY THE TOOLS.
    - Use vector search tool to retrieve appliance specific information as appliable. If you are not
    sure you can respond politely of your incognizance or you can use the youtube search tool to further
    look up if anything relevant can be obtained from the internet.
    - Use youtube search tool if user asks for video content support.
    - You can use multiple tools in chaining or ignore to use one at all.
    - Your tone must be polite, professional and helpful, but you can be witty and fun as per need.
    """
    
class AgentState(MessagesState, total=False):
    """The state of the agent."""
    

def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=instructions)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | model

async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)

    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        #If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
        
workflow = StateGraph(AgentState)

workflow.add_node("agent", acall_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

my_custom_agent = workflow.compile(checkpointer=MemorySaver())