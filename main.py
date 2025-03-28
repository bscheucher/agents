import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools.sql import run_query_tool, list_tables, describe_tables_tool
from tools.report import write_report_tool
from handlers.chat_model_start_handler import ChatModelStartHandler

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

handler = ChatModelStartHandler()
# LLM instance
chat = ChatOpenAI(
    callbacks=[handler]
)

tables = list_tables()

# Corrected prompt template
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI that has access to a SQLite database.\n"
            f"The database has tables of: {tables}\n"
            "Do not make any assumptions about what tables exist "
            "or what columns exist. Instead, use the 'describe_tables' function"
        ),
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define tools
tools = [run_query_tool, describe_tables_tool, write_report_tool]

# Create agent
agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    # verbose=True,
    tools=tools,
    memory=memory
)

# Correct invocation
response1 = agent_executor.invoke(
    {"input": "How many orders are there? Write the results to an html report."})
response2 = agent_executor.invoke(
    {"input": "Execute the same process for users."})

print(response1, "\n", response2)  # Print response for debugging
