import argparse

from app.config.settings import get_settings
from app.contracts.runtime import AgentRequest
from app.runtime.factory import create_runtime


def print_evidence(response) -> None:
    evidence = response.metadata.get("evidence") or []
    if not evidence:
        return
    print("\nEvidence:")
    for index, entry in enumerate(evidence, start=1):
        if entry.get("filename"):
            label = entry["filename"]
            if entry.get("page_number"):
                label += f", page {entry['page_number']}"
        elif entry.get("source") == "stm":
            label = "earlier conversation"
        elif entry.get("source") == "ltm":
            label = "saved memory"
        else:
            label = entry.get("source", "source")
        print(f"  [{index}] {label}")
        print(f'      "{entry["quote"]}"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--thread", required=True)
    parser.add_argument("--query")
    args = parser.parse_args()

    settings = get_settings()
    runtime = create_runtime(settings=settings)

    print("=" * 60)
    print("Conversational Memory Agent")
    print("STM + LTM RAG Agent")
    print("=" * 60)

    user_id = args.user
    thread_id = args.thread

    if args.query is not None:
        response = runtime.handle(
            AgentRequest(
                user_id=user_id,
                thread_id=thread_id,
                message=args.query,
            )
        )
        print(f"\nAssistant: {response.answer}")
        print_evidence(response)
        print(f"Trace ID: {response.trace_id}")
        return

    while True:
        message = input("\nYou: ").strip()

        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if not message:
            continue

        response = runtime.handle(
            AgentRequest(
                user_id=user_id,
                thread_id=thread_id,
                message=message,
            )
        )

        print(f"\nAssistant: {response.answer}")
        print_evidence(response)
        # print(f"Trace ID: {response.trace_id}")


if __name__ == "__main__":
    main()