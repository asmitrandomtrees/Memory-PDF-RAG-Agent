from app.config.settings import get_settings
from app.embeddings.factory import create_embedding_provider
from app.graph.rag_graph import RagGraph
from app.llm.azure_openai import AzureOpenAIProvider
from app.memory.stm.recent import RecentMessagesSTM
from app.rag.pdf_retriever import PDFRetriever
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore
from app.vectorstores.factory import create_vector_store


def create_runtime(ltm=None) -> AgentRuntime:
    settings = get_settings()
    store = JsonlConversationStore(settings.conversation_data_path)
    embedder = create_embedding_provider(settings)
    vectors = create_vector_store(settings)
    graph = RagGraph(
        stm=RecentMessagesSTM(store, keep=6),
        pdf_retriever=PDFRetriever(embedder, vectors, settings.pdf_collection, settings.pdf_top_k),
        llm=AzureOpenAIProvider(settings),
        ltm=ltm,  # plug your long-term memory here later
        max_retries=settings.max_retries,
        pdf_top_k=settings.pdf_top_k,
    )
    return AgentRuntime(conversation_store=store, graph=graph)
