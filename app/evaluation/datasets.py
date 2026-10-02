from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class EvalTestCase:
    test_id: str
    scenario: Literal[
        "stm",
        "ltm",
        "pdf",
        "stm_ltm",
        "stm_pdf",
        "ltm_pdf",
        "stm_ltm_pdf",
        "consolidation",
    ]
    query: str
    user_id: str = "user_eval_001"
    thread_id: str = "thread_eval_001"
    document_ids: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    expected_item_ids: list[str] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    expected_action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Standard benchmark test cases covering all retrieval and combination scenarios
BENCHMARK_DATASET: list[EvalTestCase] = [
    # 1. STM Scenarios
    EvalTestCase(
        test_id="stm_01_recent_thread",
        scenario="stm",
        query="What was the error message I mentioned earlier in this chat?",
        expected_sources=["stm"],
        expected_item_ids=["stm_msg_err_101"],
        expected_answer_keywords=["ConnectionTimeoutError", "port 5432"],
    ),
    EvalTestCase(
        test_id="stm_02_episodic_date_ist",
        scenario="stm",
        query="What did we plan during our discussion on September 18, 2026?",
        expected_sources=["stm"],
        expected_item_ids=["stm_msg_sep18_01"],
        expected_answer_keywords=["architecture review", "Chroma vectorstore"],
        metadata={"date_month": 9, "date_day": 18, "date_year": 2026},
    ),

    # 2. LTM Scenarios
    EvalTestCase(
        test_id="ltm_01_user_preference",
        scenario="ltm",
        query="What is my preferred programming language and code style?",
        expected_sources=["ltm"],
        expected_item_ids=["ltm_pref_001", "ltm_pref_002"],
        expected_answer_keywords=["Python", "PEP 8", "type hints"],
    ),
    EvalTestCase(
        test_id="ltm_02_user_project_fact",
        scenario="ltm",
        query="Which cloud platform do I use for deployments?",
        expected_sources=["ltm"],
        expected_item_ids=["ltm_fact_cloud_01"],
        expected_answer_keywords=["Azure", "eastus2"],
    ),

    # 3. PDF RAG Scenarios
    EvalTestCase(
        test_id="pdf_01_api_documentation",
        scenario="pdf",
        query="What are the rate limits for the batch embeddings API?",
        document_ids=["doc_api_spec_v2"],
        expected_sources=["pdf"],
        expected_item_ids=["pdf_chunk_api_lim_04"],
        expected_answer_keywords=["300 requests per minute", "1000 items per batch"],
    ),
    EvalTestCase(
        test_id="pdf_02_compliance_guideline",
        scenario="pdf",
        query="What is the data retention policy described in the compliance doc?",
        document_ids=["doc_compliance_2026"],
        expected_sources=["pdf"],
        expected_item_ids=["pdf_chunk_comp_09"],
        expected_answer_keywords=["90 days", "encrypted at rest"],
    ),

    # 4. STM + LTM Combined
    EvalTestCase(
        test_id="stm_ltm_01_preference_in_conversation",
        scenario="stm_ltm",
        query="Based on my preferred stack and what we discussed today, how should we write the database query?",
        expected_sources=["stm", "ltm"],
        expected_item_ids=["ltm_pref_001", "stm_msg_today_db_01"],
        expected_answer_keywords=["SQLAlchemy", "async session", "PostgreSQL"],
    ),

    # 5. STM + PDF Combined
    EvalTestCase(
        test_id="stm_pdf_01_troubleshooting_doc",
        scenario="stm_pdf",
        query="Given the error I just shared, what does the user manual say to fix it?",
        document_ids=["doc_troubleshooting_v1"],
        expected_sources=["stm", "pdf"],
        expected_item_ids=["stm_msg_err_101", "pdf_chunk_err_fix_02"],
        expected_answer_keywords=["restart daemon", "check firewall"],
    ),

    # 6. LTM + PDF Combined
    EvalTestCase(
        test_id="ltm_pdf_01_customized_doc_query",
        scenario="ltm_pdf",
        query="According to the company policy, what benefits apply to my role?",
        document_ids=["doc_hr_policy_2026"],
        expected_sources=["ltm", "pdf"],
        expected_item_ids=["ltm_fact_role_01", "pdf_chunk_benefits_03"],
        expected_answer_keywords=["Senior Engineer", "remote stipend"],
    ),

    # 7. STM + LTM + PDF Combined
    EvalTestCase(
        test_id="stm_ltm_pdf_01_full_hybrid",
        scenario="stm_ltm_pdf",
        query="Summarize our project goal from today, my preference, and the guidelines in the architecture PDF.",
        document_ids=["doc_arch_overview_2026"],
        expected_sources=["stm", "ltm", "pdf"],
        expected_item_ids=["stm_msg_goal_01", "ltm_pref_001", "pdf_chunk_arch_01"],
        expected_answer_keywords=["Memory RAG Agent", "Python", "Microservices"],
    ),

    # 8. Memory Consolidation Benchmark Cases
    EvalTestCase(
        test_id="cons_01_new_memory",
        scenario="consolidation",
        query="I recently started learning Rust.",
        expected_action="new",
        metadata={"user_id": "user_eval_001"},
    ),
    EvalTestCase(
        test_id="cons_02_update_memory",
        scenario="consolidation",
        query="I have now moved to Seattle instead of Austin.",
        expected_action="update",
        metadata={"user_id": "user_eval_001", "existing_id": "ltm_fact_city_01"},
    ),
    EvalTestCase(
        test_id="cons_03_supersede_memory",
        scenario="consolidation",
        query="I no longer use MongoDB; now I strictly use Chroma.",
        expected_action="supersede",
        metadata={"user_id": "user_eval_001", "existing_id": "ltm_fact_db_01"},
    ),
    EvalTestCase(
        test_id="cons_04_duplicate_memory",
        scenario="consolidation",
        query="Just a reminder, my name is Alice.",
        expected_action="duplicate",
        metadata={"user_id": "user_eval_001", "existing_id": "ltm_fact_name_01"},
    ),
]
