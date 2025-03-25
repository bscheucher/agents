import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.agents import create_openai_functions_agent, AgentExecutor
from tools.sql import run_query_tool

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# LLM instance
chat = ChatOpenAI()

# Corrected prompt template
prompt = ChatPromptTemplate(
    input_variables=["input", "agent_scratchpad"],  # Define expected input variables
    messages=[
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ]
)

# Define tools
tools = [run_query_tool]

# Create agent
agent = create_openai_functions_agent(
    llm=chat,
    prompt=prompt,
    tools=tools
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools
)

# Correct invocation
response = agent_executor.invoke({"input": "How many users are in the database?"})

print(response)  # Print response for debugging
