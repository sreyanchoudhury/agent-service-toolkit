import os
import sys
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class SafeAzureSearch(AzureSearch):
    def __del__(self):
        try:
            if "asyncio" in sys.modules:
                super().__del__()
        except (ModuleNotFoundError, RuntimeError):
            pass

embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

vector_store: AzureSearch = SafeAzureSearch(
    azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
    index_name=os.getenv("AZURE_AI_SEARCH_INDEX"),
    search_type="similarity_score_threshold",
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 3},
    kwargs={"k": 3, "score_threshold": 0.8}
)

vector_search_retreiver = vector_store.as_retriever()
vector_search_tool = create_retriever_tool(
  vector_search_retreiver,
  name="azure_ai_vector_search",
  description="Tool to retrieve results against the Vector Search index",
  response_format="content"
)

__all__ = ["vector_search_tool"]