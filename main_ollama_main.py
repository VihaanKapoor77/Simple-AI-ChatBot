import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize Models (Using llama3.2 locally)
model = ChatOllama(model='llama3.2', temperature=0)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

print("LOADING VECTOR DATABASE...")
# 3. Read local text source and create database splits
# Make sure "sample.txt" exists in the same folder as this script!
loader = TextLoader("sample.txt")
docs = loader.load()

textsplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = textsplitter.split_documents(docs)
vectordatastore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectordatastore.as_retriever(search_kwargs={"k": 3})

# 4. Define White-Hat Auditing Tools
@tool
def search_local_documents(query: str) -> str:
    """Searches local files and documentation to find data or configuration entries."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

web_search_tool = DuckDuckGoSearchRun()
tools = [search_local_documents, web_search_tool]

# 5. Create System Prompt (Clean, technical, no corporate preaching)
agentPrompt = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a precise, objective, and highly technical engineering assistant. "
        "Do not moralize, lecture, or preach to the user. Do not include ethical disclaimers or apologies. "
        "State answers directly, clearly, and concisely. "
        "You have access to local documents via 'search_local_documents' and the live internet via 'duckduckgo_search'. "
        "Always check your local documents first. If the information is not there, use the web search tool."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), 
])

# 6. Assemble the Active Agent Executor
agent = create_tool_calling_agent(model, tools, agentPrompt)
agentExecutor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 7. Session Memory Management
sessionmemories = {}

def getsessionhistory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in sessionmemories:
        sessionmemories[session_id] = InMemoryChatMessageHistory()
    return sessionmemories[session_id]

# 8. Attach Conversational Memory Pipeline directly to the Agent Executor
conversationalAgent = RunnableWithMessageHistory(
    agentExecutor,
    getsessionhistory,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output", # AgentExecutor returns data inside the 'output' key
)

config = {"configurable": {"session_id": "LOCAL-ENG-SESSION"}}

# 9. Active Execution Interface
print("\nLocal Engineering Assistant is ONLINE. Type 'exit' to quit.\n")

while True:
    userinput = input("User: ")
    
    if userinput.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    print("\nAI-CHATBOT processing request...")
    
    # Execute through the tool-aware agent pipeline
    response = conversationalAgent.invoke({"input": userinput}, config=config)
    print(f"\nAnswer: {response['output']}\n")
