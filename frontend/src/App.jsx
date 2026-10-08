import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Check,
  CheckCircle2,
  FileText,
  FolderOpen,
  LoaderCircle,
  MessageSquare,
  Plus,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  X,
} from "lucide-react";

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof payload.detail === "string"
      ? payload.detail
      : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return payload;
}

function formatTime(timestamp) {
  if (!timestamp) return "";
  return new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(timestamp));
}

function App() {
  const [threads, setThreads] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [activeThread, setActiveThread] = useState("");
  const [messages, setMessages] = useState([]);
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [uploadNotice, setUploadNotice] = useState("");
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const fileInput = useRef(null);
  const messageList = useRef(null);
  const textarea = useRef(null);

  const refreshThreads = useCallback(async () => {
    const result = await api("/api/threads");
    setThreads(result);
    return result;
  }, []);

  const refreshDocuments = useCallback(async () => {
    const result = await api("/api/documents");
    setDocuments(result);
    setSelectedDocuments((current) =>
      current.filter((id) => result.some((document) => document.document_id === id)),
    );
    return result;
  }, []);

  const makeThread = useCallback(async () => {
    const result = await api("/api/threads", { method: "POST" });
    setActiveThread(result.thread_id);
    setMessages([]);
    setMobileSidebarOpen(false);
    return result.thread_id;
  }, []);

  useEffect(() => {
    let mounted = true;
    async function initialize() {
      try {
        const [loadedThreads] = await Promise.all([
          refreshThreads(),
          refreshDocuments(),
          api("/api/health"),
        ]);
        if (!mounted) return;
        if (loadedThreads.length) {
          setActiveThread(loadedThreads[0].thread_id);
        } else {
          const created = await api("/api/threads", { method: "POST" });
          if (mounted) setActiveThread(created.thread_id);
        }
      } catch (err) {
        if (mounted) setError(err.message);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    initialize();
    return () => {
      mounted = false;
    };
  }, [refreshDocuments, refreshThreads]);

  useEffect(() => {
    if (!activeThread) return;
    let mounted = true;
    setMessages([]);
    api(`/api/threads/${encodeURIComponent(activeThread)}/messages`)
      .then((result) => {
        if (mounted) setMessages(result.messages);
      })
      .catch((err) => {
        if (mounted) setError(err.message);
      });
    return () => {
      mounted = false;
    };
  }, [activeThread]);

  useEffect(() => {
    messageList.current?.scrollTo({
      top: messageList.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, sending]);

  async function submitMessage(event) {
    event?.preventDefault();
    const content = draft.trim();
    if (!content || sending || !activeThread) return;
    setError("");
    setDraft("");
    setSending(true);
    const optimisticId = `pending-${Date.now()}`;
    setMessages((current) => [
      ...current,
      {
        message_id: optimisticId,
        role: "user",
        content,
        timestamp: new Date().toISOString(),
      },
    ]);
    try {
      await api(`/api/threads/${encodeURIComponent(activeThread)}/messages`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          document_ids: selectedDocuments,
        }),
      });
      const [threadResult] = await Promise.all([
        api(`/api/threads/${encodeURIComponent(activeThread)}/messages`),
        refreshThreads(),
      ]);
      setMessages(threadResult.messages);
    } catch (err) {
      setError(err.message);
      try {
        const threadResult = await api(
          `/api/threads/${encodeURIComponent(activeThread)}/messages`,
        );
        setMessages(threadResult.messages);
      } catch {
        setMessages((current) =>
          current.filter((message) => message.message_id !== optimisticId),
        );
      }
    } finally {
      setSending(false);
      textarea.current?.focus();
    }
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return;
    setUploading(true);
    setError("");
    setUploadNotice("");
    let uploaded = 0;
    try {
      for (const file of files) {
        const form = new FormData();
        form.append("file", file);
        await api("/api/documents", { method: "POST", body: form });
        uploaded += 1;
      }
      await refreshDocuments();
      setUploadNotice(
        `${uploaded} PDF${uploaded === 1 ? "" : "s"} added to your knowledge base.`,
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function removeDocument(documentId) {
    setError("");
    try {
      await api(`/api/documents/${encodeURIComponent(documentId)}`, {
        method: "DELETE",
      });
      await refreshDocuments();
    } catch (err) {
      setError(err.message);
    }
  }

  function toggleDocument(documentId) {
    setSelectedDocuments((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
  }

  const activeTitle =
    threads.find((thread) => thread.thread_id === activeThread)?.title ||
    "New conversation";
  const hasMessages = messages.length > 0;

  return (
    <div className="flex h-dvh overflow-hidden bg-[#f8f9fb] text-[#172126]">
      {mobileSidebarOpen && (
        <button
          className="fixed inset-0 z-30 bg-slate-950/30 lg:hidden"
          aria-label="Close navigation"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[292px] shrink-0 flex-col border-r border-[#e9eded] bg-white transition-transform lg:static lg:translate-x-0 ${
          mobileSidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex h-[76px] items-center justify-between border-b border-[#eef0f0] px-5">
          <a href="/" className="flex items-center gap-3" aria-label="Memo home">
            <span className="grid size-10 place-items-center rounded-[14px] bg-[#d9f5ef] text-[#087d70]">
              <Sparkles size={20} strokeWidth={2.1} />
            </span>
            <span>
              <span className="block text-[17px] font-bold tracking-[-0.04em]">memo</span>
              <span className="block text-[10px] font-semibold uppercase tracking-[0.15em] text-[#98a3a4]">
                memory assistant
              </span>
            </span>
          </a>
          <button
            className="grid size-9 place-items-center rounded-lg text-slate-400 hover:bg-slate-50 lg:hidden"
            aria-label="Close sidebar"
            onClick={() => setMobileSidebarOpen(false)}
          >
            <X size={18} />
          </button>
        </div>

        <div className="px-4 pt-5">
          <button
            onClick={makeThread}
            className="flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#117d70] text-sm font-semibold text-white shadow-sm transition hover:bg-[#096b60]"
          >
            <Plus size={17} />
            New conversation
          </button>
        </div>

        <section className="scrollbar-subtle mt-7 flex-1 overflow-y-auto px-3">
          <div className="mb-2 flex items-center justify-between px-2">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#9ca7a8]">
              Recent conversations
            </h2>
            <span className="rounded-md bg-[#f1f4f4] px-1.5 py-0.5 text-[10px] font-semibold text-[#8b9697]">
              {threads.length}
            </span>
          </div>
          <div className="space-y-1">
            {threads.map((thread) => (
              <button
                key={thread.thread_id}
                onClick={() => {
                  setActiveThread(thread.thread_id);
                  setMobileSidebarOpen(false);
                }}
                className={`flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2.5 text-left transition ${
                  activeThread === thread.thread_id
                    ? "bg-[#edf7f5] text-[#126e64]"
                    : "text-[#687577] hover:bg-[#f6f8f8]"
                }`}
              >
                <MessageSquare size={15} className="mt-0.5 shrink-0" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[12px] font-medium">
                    {thread.title}
                  </span>
                  <span className="mt-1 block text-[10px] text-[#a0aaaa]">
                    {formatTime(thread.updated_at)}
                  </span>
                </span>
              </button>
            ))}
            {!threads.length && !loading && (
              <p className="px-2.5 py-3 text-xs text-[#9aa5a5]">
                Your conversations will appear here.
              </p>
            )}
          </div>

          <div className="mb-2 mt-8 flex items-center justify-between px-2">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#9ca7a8]">
              Knowledge base
            </h2>
            <span className="rounded-md bg-[#f1f4f4] px-1.5 py-0.5 text-[10px] font-semibold text-[#8b9697]">
              {documents.length}
            </span>
          </div>

          <div className="mb-3 rounded-xl border border-dashed border-[#d9e2e1] bg-[#fbfcfc] p-3">
            <input
              ref={fileInput}
              className="hidden"
              type="file"
              accept="application/pdf,.pdf"
              multiple
              onChange={(event) => uploadFiles(event.target.files)}
            />
            <button
              onClick={() => fileInput.current?.click()}
              disabled={uploading}
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#dfe8e6] bg-white px-3 py-2 text-xs font-semibold text-[#45605c] shadow-[0_1px_2px_rgb(0_0_0_0.03)] transition hover:border-[#9ecfc4] hover:text-[#087d70] disabled:opacity-60"
            >
              {uploading ? (
                <LoaderCircle size={15} className="animate-spin" />
              ) : (
                <Upload size={15} />
              )}
              {uploading ? "Adding PDFs..." : "Add PDF documents"}
            </button>
            <p className="mt-2 text-center text-[10px] leading-4 text-[#9aa5a5]">
              PDF only · up to 25 MB each
            </p>
          </div>

          <div className="space-y-1 pb-5">
            {documents.map((document) => {
              const selected = selectedDocuments.includes(document.document_id);
              return (
                <div
                  key={document.document_id}
                  className={`group flex items-center gap-2 rounded-lg px-2 py-2 ${
                    selected ? "bg-[#f0f8f6]" : "hover:bg-[#f6f8f8]"
                  }`}
                >
                  <button
                    onClick={() => toggleDocument(document.document_id)}
                    className="flex min-w-0 flex-1 items-center gap-2.5 text-left"
                    title={
                      selected
                        ? "Limit search to this PDF"
                        : "Include this PDF in every search"
                    }
                  >
                    <span
                      className={`grid size-7 shrink-0 place-items-center rounded-lg ${
                        selected
                          ? "bg-[#d9f5ef] text-[#087d70]"
                          : "bg-[#fff2ed] text-[#ce7257]"
                      }`}
                    >
                      {selected ? <Check size={14} /> : <FileText size={14} />}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[11px] font-medium text-[#4e5a5c]">
                        {document.filename}
                      </span>
                      <span className="block text-[10px] text-[#9ca7a8]">
                        {document.chunks} text chunks
                      </span>
                    </span>
                  </button>
                  <button
                    onClick={() => removeDocument(document.document_id)}
                    className="grid size-7 shrink-0 place-items-center rounded-md text-[#b4bcbc] opacity-0 transition hover:bg-red-50 hover:text-red-500 group-hover:opacity-100 focus:opacity-100"
                    title={`Remove ${document.filename}`}
                    aria-label={`Remove ${document.filename}`}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              );
            })}
            {!documents.length && !loading && (
              <div className="px-2 py-2 text-[11px] leading-5 text-[#9aa5a5]">
                Upload a PDF to ask questions grounded in your documents.
              </div>
            )}
          </div>
        </section>

        <div className="border-t border-[#eef0f0] p-4">
          <div className="flex items-center gap-2.5 rounded-xl bg-[#f7f9f9] p-3">
            <span className="grid size-8 place-items-center rounded-full bg-[#dceeea] text-[11px] font-bold text-[#28796c]">
              M
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-xs font-semibold text-[#394446]">Local workspace</span>
              <span className="mt-0.5 block text-[10px] text-[#929d9e]">Private on this machine</span>
            </span>
            <ShieldCheck size={16} className="text-[#7da99e]" />
          </div>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-[#e9eded] bg-white/85 px-4 backdrop-blur sm:px-7">
          <div className="flex min-w-0 items-center gap-3">
            <button
              className="grid size-9 place-items-center rounded-lg text-[#667476] hover:bg-[#f3f6f6] lg:hidden"
              aria-label="Open navigation"
              onClick={() => setMobileSidebarOpen(true)}
            >
              <FolderOpen size={18} />
            </button>
            <span className="min-w-0">
              <span className="block truncate text-[13px] font-semibold text-[#263234]">
                {activeTitle}
              </span>
              <span className="mt-1 flex items-center gap-1.5 text-[10px] text-[#96a1a2]">
                <span className="size-1.5 rounded-full bg-[#46b28d]" />
                STM · LTM · PDF RAG
              </span>
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-[#e9eeee] bg-white px-3 py-1.5 text-[10px] font-medium text-[#819091]">
            <CheckCircle2 size={13} className="text-[#53a888]" />
            Local agent
          </div>
        </header>

        <div
          ref={messageList}
          className="chat-scroll scrollbar-subtle flex-1 overflow-y-auto"
        >
          {error && (
            <div className="mx-auto mt-5 flex max-w-3xl items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <span>{error}</span>
              <button
                className="shrink-0 rounded p-0.5 hover:bg-red-100"
                aria-label="Dismiss error"
                onClick={() => setError("")}
              >
                <X size={16} />
              </button>
            </div>
          )}
          {uploadNotice && (
            <div className="mx-auto mt-5 flex max-w-3xl items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
              <CheckCircle2 size={16} />
              {uploadNotice}
              <button
                className="ml-auto"
                aria-label="Dismiss notification"
                onClick={() => setUploadNotice("")}
              >
                <X size={15} />
              </button>
            </div>
          )}
          {loading ? (
            <div className="grid h-full place-items-center text-sm text-[#8b9899]">
              <span className="flex items-center gap-2">
                <LoaderCircle size={18} className="animate-spin" />
                Connecting to your local agent...
              </span>
            </div>
          ) : !hasMessages ? (
            <Welcome
              onPrompt={(prompt) => {
                setDraft(prompt);
                textarea.current?.focus();
              }}
              hasDocuments={documents.length > 0}
            />
          ) : (
            <div className="mx-auto max-w-3xl px-4 pb-10 pt-8 sm:px-8 sm:pt-12">
              <div className="mb-8 flex justify-center">
                <span className="rounded-full border border-[#e9eeee] bg-white px-3 py-1 text-[10px] font-medium text-[#9aa5a5]">
                  Conversation history
                </span>
              </div>
              <div className="space-y-7">
                {messages.map((message) => (
                  <Message key={message.message_id} message={message} />
                ))}
                {sending && (
                  <div className="animate-float-in flex items-start gap-3">
                    <AssistantAvatar />
                    <div className="rounded-2xl rounded-tl-md border border-[#e9eeee] bg-white px-4 py-3 text-[#829091] shadow-[0_2px_8px_rgb(26_49_47_0.025)]">
                      <span className="flex items-center gap-2 text-xs">
                        <LoaderCircle size={14} className="animate-spin text-[#159180]" />
                        Thinking with your context...
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="shrink-0 bg-gradient-to-t from-[#f8f9fb] via-[#f8f9fb] to-transparent px-3 pb-3 pt-2 sm:px-6 sm:pb-6">
          <form
            onSubmit={submitMessage}
            className="mx-auto max-w-3xl rounded-[20px] border border-[#e3e9e8] bg-white p-2 shadow-[0_8px_30px_rgb(24_54_50_0.07)] transition focus-within:border-[#add8cf] focus-within:shadow-[0_8px_34px_rgb(24_100_85_0.10)]"
          >
            {selectedDocuments.length > 0 && (
              <div className="flex flex-wrap gap-1.5 px-2 pb-1 pt-1">
                {documents
                  .filter((document) => selectedDocuments.includes(document.document_id))
                  .map((document) => (
                    <button
                      type="button"
                      key={document.document_id}
                      onClick={() => toggleDocument(document.document_id)}
                      className="inline-flex max-w-full items-center gap-1 rounded-md bg-[#edf7f5] px-2 py-1 text-[10px] font-medium text-[#32786e]"
                      title="Remove document filter"
                    >
                      <FileText size={11} />
                      <span className="max-w-48 truncate">{document.filename}</span>
                      <X size={11} />
                    </button>
                  ))}
              </div>
            )}
            <textarea
              ref={textarea}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submitMessage();
                }
              }}
              placeholder="Ask about your conversation or a PDF..."
              rows={1}
              maxLength={12000}
              disabled={loading || sending || !activeThread}
              className="max-h-40 min-h-12 w-full resize-none bg-transparent px-3 py-3 text-[13px] leading-5 text-[#263234] outline-none placeholder:text-[#a5afaf] disabled:opacity-60"
            />
            <div className="flex items-center justify-between px-1 pb-0.5">
              <div className="flex items-center gap-2 pl-2 text-[10px] text-[#a0aaaa]">
                <span className="flex items-center gap-1">
                  <ShieldCheck size={12} />
                  Your data stays local
                </span>
                {selectedDocuments.length === 0 && documents.length > 0 && (
                  <span className="hidden sm:inline">· Searching all PDFs</span>
                )}
              </div>
              <button
                type="submit"
                disabled={!draft.trim() || sending || loading}
                className="grid size-9 place-items-center rounded-xl bg-[#117d70] text-white transition hover:bg-[#096b60] disabled:cursor-not-allowed disabled:bg-[#d9e3e1] disabled:text-[#9eaaa9]"
                aria-label="Send message"
              >
                {sending ? (
                  <LoaderCircle size={16} className="animate-spin" />
                ) : (
                  <ArrowUp size={17} strokeWidth={2.4} />
                )}
              </button>
            </div>
          </form>
          <p className="mx-auto mt-2 max-w-3xl text-center text-[9px] text-[#aab3b3]">
            Answers may be imperfect. Verify important information against its source.
          </p>
        </div>
      </main>
    </div>
  );
}

function AssistantAvatar() {
  return (
    <span className="grid size-8 shrink-0 place-items-center rounded-[11px] bg-[#d9f5ef] text-[#087d70]">
      <Sparkles size={15} />
    </span>
  );
}

function Message({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={`animate-float-in flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {isUser ? (
        <span className="grid size-8 shrink-0 place-items-center rounded-full bg-[#e9ecec] text-[10px] font-bold text-[#657172]">
          You
        </span>
      ) : (
        <AssistantAvatar />
      )}
      <div className={`max-w-[85%] sm:max-w-[78%] ${isUser ? "text-right" : ""}`}>
        <div
          className={`inline-block rounded-2xl px-4 py-3 text-left text-[13px] leading-[1.75] ${
            isUser
              ? "rounded-tr-md bg-[#e8f3f1] text-[#293b39]"
              : "rounded-tl-md border border-[#e9eeee] bg-white text-[#374345] shadow-[0_2px_8px_rgb(26_49_47_0.025)]"
          }`}
        >
          <div className="message-content whitespace-pre-wrap break-words">
            {message.content}
          </div>
        </div>
        {!isUser && Array.isArray(message.metadata?.citations) && message.metadata.citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {message.metadata.citations.map((citation, index) => (
              <span
                key={`${citation.item_id || citation.document_id || "source"}-${index}`}
                className="inline-flex items-center gap-1 rounded-md border border-[#e8edec] bg-white px-2 py-1 text-[9px] text-[#839091]"
              >
                <FileText size={10} />
                {citation.filename || citation.source || "Source"}
                {citation.page ? ` · p.${citation.page}` : ""}
              </span>
            ))}
          </div>
        )}
        <div className={`mt-1.5 text-[9px] text-[#aab3b3] ${isUser ? "pr-1" : "pl-1"}`}>
          {isUser ? "You" : "Memo"} · {formatTime(message.timestamp)}
        </div>
      </div>
    </div>
  );
}

function Welcome({ onPrompt, hasDocuments }) {
  const prompts = [
    {
      title: "Recall a detail",
      text: "What did we discuss at the start of this conversation?",
      icon: <MessageSquare size={15} />,
    },
    {
      title: "Ask your documents",
      text: "What are the main points in my PDFs?",
      icon: <FileText size={15} />,
    },
    {
      title: "Connect the dots",
      text: "How does my project relate to the uploaded documents?",
      icon: <Sparkles size={15} />,
    },
  ];
  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col justify-center px-5 pb-10 pt-12 sm:px-8">
      <div className="animate-float-in mb-8">
        <span className="mb-5 inline-flex items-center gap-1.5 rounded-full border border-[#dfeae7] bg-white px-3 py-1.5 text-[10px] font-semibold text-[#4f8b7e] shadow-[0_1px_4px_rgb(0_0_0_0.025)]">
          <Sparkles size={12} />
          Your personal memory assistant
        </span>
        <h1 className="max-w-xl text-[34px] font-semibold leading-[1.13] tracking-[-0.055em] text-[#1e2c2e] sm:text-[43px]">
          A little more context.
          <br />
          <span className="text-[#188677]">A lot more helpful.</span>
        </h1>
        <p className="mt-4 max-w-lg text-[13px] leading-6 text-[#859192]">
          Chat naturally. Memo remembers your conversation and can find answers
          in the PDFs you add to your knowledge base.
        </p>
      </div>
      <div className="grid gap-2.5 sm:grid-cols-3">
        {prompts.map((prompt, index) => (
          <button
            key={prompt.title}
            onClick={() => onPrompt(prompt.text)}
            className="animate-float-in group rounded-2xl border border-[#e7eceb] bg-white p-4 text-left shadow-[0_2px_10px_rgb(26_49_47_0.025)] transition hover:-translate-y-0.5 hover:border-[#b9d9d2] hover:shadow-[0_8px_20px_rgb(26_49_47_0.07)]"
            style={{ animationDelay: `${index * 60}ms` }}
          >
            <span className="mb-4 grid size-8 place-items-center rounded-[10px] bg-[#eef7f5] text-[#248476] transition group-hover:bg-[#d9f5ef]">
              {prompt.icon}
            </span>
            <span className="block text-[11px] font-semibold text-[#405052]">
              {prompt.title}
            </span>
            <span className="mt-1.5 block text-[10px] leading-4 text-[#929e9f]">
              {prompt.title === "Ask your documents" && !hasDocuments
                ? "Add a PDF, then ask away"
                : prompt.text}
            </span>
          </button>
        ))}
      </div>
      <div className="mt-7 flex items-center gap-2 text-[10px] text-[#a0aaaa]">
        <ArrowDown size={12} />
        Start with a question or choose a prompt
      </div>
    </div>
  );
}

export default App;
