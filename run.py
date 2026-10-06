from app.contracts.runtime import AgentRequest
from app.main import create_runtime


def main() -> None:
    runtime = create_runtime()

    print("=" * 60)
    print("Conversational Memory Agent  (STM + PDF RAG)")
    print("=" * 60)

    user_id = input("User ID: ").strip()
    thread_id = input("Thread ID: ").strip()

    while True:
        message = input("\nYou: ").strip()
        if message.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not message:
            continue

        response = runtime.handle(
            AgentRequest(user_id=user_id, thread_id=thread_id, message=message)
        )
        print(f"\nAssistant: {response.answer}")
        for s in response.metadata.get("pdf_sources", [])[:3]:
            print(f"   source: {s['file']} p.{s['page']}")
        print(f"Trace ID: {response.trace_id}")


if __name__ == "__main__":
    main()
