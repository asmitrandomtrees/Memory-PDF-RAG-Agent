from typing import Any

from app.config.settings import Settings, get_settings
from app.contracts.runtime import ConversationStore
from app.embeddings.factory import create_embedding_provider
from app.embeddings.provider import EmbeddingProvider
from app.graph.graph import RetrievalPipeline
from app.llm.azure_openai import AzureOpenAIProvider
from app.llm.provider import LLMProvider
from app.memory.ltm.consolidator import LTMConsolidator
from app.memory.ltm.extractor import LTMExtractor
from app.memory.ltm.manager import LTMManager
from app.memory.ltm.related import LTMRelatedMemoryRetriever
from app.memory.ltm.retriever import LTMRetriever
from app.memory.ltm.store import JsonMemoryStore
from app.memory.ltm.validator import LTMValidator
from app.memory.ltm.writer import LTMWriter
from app.memory.stm.episodic_retriever import STMEpisodicRetriever
from app.memory.stm.retriever import STMRetriever
from app.memory.stm.writer import STMWriter
from app.graph.reranker import Reranker
from app.runtime.agent import AgentRuntime
from app.runtime.conversation_store import JsonlConversationStore
from app.tracing.exporters import (
    CompositeTraceExporter,
    JsonlTraceExporter,
    TextTraceExporter,
)
from app.vectorstores.chroma import ChromaVectorStore
from app.vectorstores.factory import create_vector_store


def create_runtime(
    *,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    embeddings: EmbeddingProvider | None = None,
    vector_store: ChromaVectorStore | None = None,
    conversation_store: ConversationStore | None = None,
    pdf_retriever: Any | None = None,
    reranker: Reranker | None = None,
) -> AgentRuntime:
    settings = settings or get_settings()
    embeddings = embeddings or create_embedding_provider(settings)
    vector_store = vector_store or create_vector_store(settings)
    llm = llm or AzureOpenAIProvider()
    conversation_store = conversation_store or JsonlConversationStore(
        settings.conversation_data_path
    )

    stm_writer = STMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    stm_retriever = STMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    episodic_retriever = STMEpisodicRetriever(
        vector_store=vector_store,
    )

    memory_store = JsonMemoryStore(settings.ltm_memory_path)
    ltm_writer = LTMWriter(
        embeddings=embeddings,
        vector_store=vector_store,
    )
    related_retriever = LTMRelatedMemoryRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        memory_store=memory_store,
    )
    ltm_extractor = LTMExtractor(llm=llm)
    ltm_manager = LTMManager(
        memory_store=memory_store,
        validator=LTMValidator(),
        related_retriever=related_retriever,
        consolidator=LTMConsolidator(llm=llm),
        writer=ltm_writer,
    )
    ltm_retriever = LTMRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
    )

    graph = RetrievalPipeline(
        llm=llm,
        stm_retriever=stm_retriever,
        episodic_retriever=episodic_retriever,
        conversation_store=conversation_store,
        ltm_retriever=ltm_retriever,
        pdf_retriever=pdf_retriever,
        reranker=reranker,
        stm_top_k=settings.stm_top_k,
        ltm_top_k=settings.ltm_top_k,
        pdf_top_k=settings.pdf_top_k,
        max_context_tokens=settings.max_context_tokens,
        max_retries=settings.max_retries,
        trace_sink=CompositeTraceExporter(
            [
                JsonlTraceExporter(settings.trace_data_path),
                TextTraceExporter(settings.trace_data_path),
            ]
        ),
    )

    return AgentRuntime(
        conversation_store=conversation_store,
        graph=graph,
        stm_writer=stm_writer,
        ltm_extractor=ltm_extractor,
        ltm_manager=ltm_manager,
    )
