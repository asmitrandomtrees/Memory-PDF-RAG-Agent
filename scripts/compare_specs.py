"""Compare cooling tower specifications, vendor proposals or project documents.

Usage:
  python scripts/compare_specs.py --list
  python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf VendorB.pdf
  python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf --aspects fan_motor
  python scripts/compare_specs.py --docs Spec.pdf VendorA.pdf --out matrix.csv

Documents can be given as an indexed filename, a document ID, or a path to
a PDF (which is indexed first). The first document (or --baseline) is the
reference that the others are checked against. Aspects are the built-in
keys below or any free text, e.g. --aspects fan_motor "basin materials".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.comparison import (
    COMPARISON_TIMEOUT_SECONDS,
    DEFAULT_ASPECTS,
    ComparedDocument,
    SpecComparator,
    to_csv,
    to_markdown,
)
from app.config.settings import get_settings
from app.embeddings.factory import create_embedding_provider
from app.llm.azure_openai import AzureOpenAIProvider
from app.rag.pdf_ingest import document_id_for, ingest_pdf
from app.vectorstores.factory import create_vector_store


def indexed_documents(store, collection: str) -> dict[str, ComparedDocument]:
    documents: dict[str, ComparedDocument] = {}
    for item in store.get_by_metadata(collection=collection, filters={"source": "pdf"}):
        document_id = str(item.metadata.get("document_id", ""))
        if document_id and document_id not in documents:
            documents[document_id] = ComparedDocument(
                document_id=document_id,
                filename=str(item.metadata.get("filename") or document_id),
            )
    return documents


def resolve_document(name: str, known: dict[str, ComparedDocument], ingest) -> ComparedDocument:
    if name in known:
        return known[name]
    path = Path(name)
    if path.suffix.lower() == ".pdf" and path.is_file():
        document_id = document_id_for(path)
        if document_id not in known:
            print(f"Indexing {path} ...", file=sys.stderr)
            if not ingest(path):
                raise SystemExit(f"No text could be extracted from {path}")
        known[document_id] = ComparedDocument(document_id=document_id, filename=path.name)
        return known[document_id]
    matches = [d for d in known.values() if d.filename.lower() == name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"'{name}' matches several documents; use its document ID")
    raise SystemExit(f"Document not found: {name} (run with --list to see indexed documents)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Built-in aspects: " + ", ".join(a.key for a in DEFAULT_ASPECTS),
    )
    parser.add_argument("--docs", nargs="+", help="Documents to compare (2 or more)")
    parser.add_argument("--baseline", help="Reference document (default: first of --docs)")
    parser.add_argument("--aspects", nargs="+", help="Aspects to compare (default: all built-in)")
    parser.add_argument("--format", choices=["md", "csv", "json"], help="Output format (default: from --out extension, else md)")
    parser.add_argument("--out", help="Write the report to this file instead of stdout")
    parser.add_argument("--list", action="store_true", help="List indexed documents and exit")
    args = parser.parse_args()

    settings = get_settings()
    store = create_vector_store(settings)
    known = indexed_documents(store, settings.pdf_collection)

    if args.list:
        for document in known.values():
            print(f"{document.filename}\t{document.document_id}")
        if not known:
            print("No documents indexed. Run scripts/ingest_pdfs.py or pass PDF paths to --docs.")
        return

    if not args.docs or len(args.docs) < 2:
        parser.error("--docs needs at least two documents")

    embeddings = create_embedding_provider(settings)

    def ingest(path: Path) -> int:
        return ingest_pdf(path, embeddings, store, settings.pdf_collection)

    documents = [resolve_document(name, known, ingest) for name in args.docs]
    baseline = resolve_document(args.baseline, known, ingest) if args.baseline else documents[0]
    if baseline.document_id not in {d.document_id for d in documents}:
        documents.insert(0, baseline)

    comparator = SpecComparator(
        llm=AzureOpenAIProvider(timeout_seconds=COMPARISON_TIMEOUT_SECONDS),
        embeddings=embeddings,
        vector_store=store,
        collection=settings.pdf_collection,
    )
    print(
        f"Comparing {len(documents)} documents against {baseline.filename} ...",
        file=sys.stderr,
    )
    report = comparator.compare(
        documents,
        baseline_document_id=baseline.document_id,
        aspects=args.aspects,
    )

    output_format = args.format or (
        Path(args.out).suffix.lstrip(".").lower() if args.out else "md"
    )
    if output_format == "csv":
        text = to_csv(report)
    elif output_format == "json":
        text = report.model_dump_json(indent=2)
    else:
        text = to_markdown(report)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig so Excel opens the CSV with the right encoding.
        Path(args.out).write_text(
            text,
            encoding="utf-8-sig" if output_format == "csv" else "utf-8",
            newline="",
        )
        summary = report.summary
        print(
            f"Wrote {args.out}: {summary['total']} parameters, "
            f"{summary['deviation']} deviations, {summary['missing']} missing.",
            file=sys.stderr,
        )
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()