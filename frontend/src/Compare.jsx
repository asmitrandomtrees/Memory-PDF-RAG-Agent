import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  Check,
  CircleCheck,
  CircleMinus,
  Download,
  FileText,
  GitCompareArrows,
  Info,
  LoaderCircle,
  Plus,
  Scale,
  TriangleAlert,
  X,
} from "lucide-react";
import { API_BASE_URL, api } from "./api.js";

const STATUS = {
  match: { label: "Match", tone: "bg-[#e6f6ef] text-[#1f7a55]", icon: CircleCheck },
  deviation: { label: "Deviation", tone: "bg-[#fdecea] text-[#b4372b]", icon: TriangleAlert },
  missing: { label: "Missing", tone: "bg-[#fff4e0] text-[#a5650b]", icon: CircleMinus },
  not_in_baseline: { label: "Not in baseline", tone: "bg-[#eef1f5] text-[#5b6675]", icon: Info },
};

const ASSESSMENT = {
  baseline: { label: "Baseline", tone: "text-[#5b6675]" },
  compliant: { label: "Compliant", tone: "text-[#1f7a55]" },
  deviation: { label: "Deviation", tone: "text-[#b4372b]" },
  not_specified: { label: "Not specified", tone: "text-[#a5650b]" },
  no_requirement: { label: "No requirement", tone: "text-[#5b6675]" },
};

const FILTERS = ["all", "deviation", "missing", "not_in_baseline", "match"];

function guessBaseline(documents) {
  const spec = documents.find((d) => /spec|requirement|project/i.test(d.filename));
  return (spec || documents[0])?.document_id || "";
}

export default function ComparePanel({ documents, onClose }) {
  const [selected, setSelected] = useState(() => documents.map((d) => d.document_id));
  const [baseline, setBaseline] = useState(() => guessBaseline(documents));
  const [aspects, setAspects] = useState([]);
  const [chosenAspects, setChosenAspects] = useState([]);
  const [customAspect, setCustomAspect] = useState("");
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    api("/api/compare/aspects")
      .then((result) => {
        setAspects(result);
        setChosenAspects(result.map((a) => a.key));
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selected.includes(baseline) && selected.length) setBaseline(selected[0]);
  }, [selected, baseline]);

  function toggle(list, setList, value) {
    setList(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);
  }

  function addCustomAspect(event) {
    event.preventDefault();
    const text = customAspect.trim();
    if (!text) return;
    setAspects((current) =>
      current.some((a) => a.key === text) ? current : [...current, { key: text, title: text, custom: true }],
    );
    setChosenAspects((current) => (current.includes(text) ? current : [...current, text]));
    setCustomAspect("");
  }

  async function runComparison() {
    setError("");
    setRunning(true);
    setReport(null);
    setFilter("all");
    try {
      const result = await api("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_ids: [baseline, ...selected.filter((id) => id !== baseline)],
          baseline_document_id: baseline,
          aspects: chosenAspects,
        }),
      });
      setReport(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setRunning(false);
    }
  }

  async function download(format) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/compare/export?format=${format}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(report),
      });
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = format === "csv" ? "compliance-matrix.csv" : "spec-comparison.md";
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    }
  }

  const canRun = selected.length >= 2 && chosenAspects.length > 0 && baseline && !running;

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-[#f8f9fb]">
      <header className="flex h-16 shrink-0 items-center justify-between gap-3 border-b border-[#e9eded] bg-white px-4 sm:px-7">
        <div className="flex min-w-0 items-center gap-3">
          <button
            onClick={onClose}
            className="grid size-9 shrink-0 place-items-center rounded-lg text-[#667476] hover:bg-[#f3f6f6]"
            aria-label="Back to chat"
          >
            <ArrowLeft size={18} />
          </button>
          <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-[#d9f5ef] text-[#087d70]">
            <GitCompareArrows size={17} />
          </span>
          <span className="min-w-0">
            <span className="block truncate text-[14px] font-semibold text-[#263234]">
              Specification comparator
            </span>
            <span className="block truncate text-[10px] text-[#96a1a2]">
              Compare requirements across specs, vendor proposals and project documents
            </span>
          </span>
        </div>
        {report && (
          <div className="flex shrink-0 gap-2">
            <button
              onClick={() => download("csv")}
              className="flex items-center gap-1.5 rounded-lg border border-[#dfe8e6] bg-white px-3 py-2 text-xs font-semibold text-[#45605c] hover:border-[#9ecfc4] hover:text-[#087d70]"
            >
              <Download size={14} />
              <span className="hidden sm:inline">Matrix</span> CSV
            </button>
            <button
              onClick={() => download("md")}
              className="flex items-center gap-1.5 rounded-lg border border-[#dfe8e6] bg-white px-3 py-2 text-xs font-semibold text-[#45605c] hover:border-[#9ecfc4] hover:text-[#087d70]"
            >
              <Download size={14} />
              <span className="hidden sm:inline">Report</span> MD
            </button>
          </div>
        )}
      </header>

      <div className="scrollbar-subtle flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-4 py-6 sm:px-7">
          {error && (
            <div className="mb-5 flex items-start justify-between gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              <span>{error}</span>
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}

          <section className="grid gap-4 md:grid-cols-2">
            <div className="rounded-2xl border border-[#e9eeee] bg-white p-4">
              <h3 className="text-[11px] font-bold uppercase tracking-[0.12em] text-[#9ca7a8]">
                Documents
              </h3>
              <p className="mt-1 text-[11px] text-[#8b9697]">
                Pick two or more. The baseline holds the project requirements the others are checked against.
              </p>
              <div className="mt-3 space-y-1.5">
                {documents.map((doc) => {
                  const isSelected = selected.includes(doc.document_id);
                  const isBaseline = baseline === doc.document_id;
                  return (
                    <div
                      key={doc.document_id}
                      className={`flex items-center gap-2 rounded-lg border px-2.5 py-2 ${
                        isSelected ? "border-[#cfe9e3] bg-[#f4fbf9]" : "border-[#eef1f1]"
                      }`}
                    >
                      <button
                        onClick={() => toggle(selected, setSelected, doc.document_id)}
                        className="flex min-w-0 flex-1 items-center gap-2 text-left"
                        aria-pressed={isSelected}
                      >
                        <span
                          className={`grid size-5 shrink-0 place-items-center rounded-md border ${
                            isSelected ? "border-[#117d70] bg-[#117d70] text-white" : "border-[#cfd8d7] bg-white"
                          }`}
                        >
                          {isSelected && <Check size={12} />}
                        </span>
                        <FileText size={14} className="shrink-0 text-[#ce7257]" />
                        <span className="truncate text-[12px] font-medium text-[#4e5a5c]">{doc.filename}</span>
                      </button>
                      <button
                        onClick={() => {
                          if (!isSelected) setSelected([...selected, doc.document_id]);
                          setBaseline(doc.document_id);
                        }}
                        className={`flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[10px] font-semibold ${
                          isBaseline
                            ? "bg-[#117d70] text-white"
                            : "text-[#8b9697] hover:bg-[#eef4f3] hover:text-[#087d70]"
                        }`}
                        title="Use as baseline (project requirements)"
                      >
                        <Scale size={11} />
                        {isBaseline ? "Baseline" : "Set baseline"}
                      </button>
                    </div>
                  );
                })}
                {documents.length < 2 && (
                  <p className="rounded-lg bg-[#fff4e0] px-3 py-2 text-[11px] text-[#a5650b]">
                    Upload at least two PDFs to compare.
                  </p>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-[#e9eeee] bg-white p-4">
              <h3 className="text-[11px] font-bold uppercase tracking-[0.12em] text-[#9ca7a8]">
                What to compare
              </h3>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {aspects.map((aspect) => {
                  const on = chosenAspects.includes(aspect.key);
                  return (
                    <button
                      key={aspect.key}
                      onClick={() => toggle(chosenAspects, setChosenAspects, aspect.key)}
                      aria-pressed={on}
                      className={`flex items-center gap-1 rounded-full border px-3 py-1.5 text-[11px] font-medium ${
                        on
                          ? "border-[#117d70] bg-[#117d70] text-white"
                          : "border-[#dfe8e6] bg-white text-[#5b6a6c] hover:border-[#9ecfc4]"
                      }`}
                    >
                      {on && <Check size={11} />}
                      {aspect.title}
                    </button>
                  );
                })}
              </div>
              <form onSubmit={addCustomAspect} className="mt-3 flex gap-2">
                <input
                  value={customAspect}
                  onChange={(e) => setCustomAspect(e.target.value)}
                  placeholder="Add a topic, e.g. basin materials"
                  className="min-w-0 flex-1 rounded-lg border border-[#dfe8e6] px-3 py-2 text-[12px] outline-none focus:border-[#9ecfc4]"
                />
                <button
                  type="submit"
                  className="grid size-9 shrink-0 place-items-center rounded-lg border border-[#dfe8e6] text-[#45605c] hover:border-[#9ecfc4]"
                  aria-label="Add topic"
                >
                  <Plus size={15} />
                </button>
              </form>
              <button
                onClick={runComparison}
                disabled={!canRun}
                className="mt-4 flex h-11 w-full items-center justify-center gap-2 rounded-xl bg-[#117d70] text-sm font-semibold text-white shadow-sm hover:bg-[#096b60] disabled:opacity-50"
              >
                {running ? <LoaderCircle size={16} className="animate-spin" /> : <GitCompareArrows size={16} />}
                {running ? "Comparing… this can take a minute" : "Compare documents"}
              </button>
            </div>
          </section>

          {report && <Report report={report} filter={filter} setFilter={setFilter} />}
        </div>
      </div>
    </div>
  );
}

function Report({ report, filter, setFilter }) {
  const names = useMemo(
    () => Object.fromEntries(report.documents.map((d) => [d.document_id, d.filename])),
    [report],
  );
  const summary = report.summary;

  return (
    <div className="mt-6 space-y-5">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
        {FILTERS.map((key) => {
          const meta = STATUS[key];
          const count = key === "all" ? summary.total : summary[key];
          const Icon = meta?.icon;
          return (
            <button
              key={key}
              onClick={() => setFilter(key)}
              aria-pressed={filter === key}
              className={`rounded-xl border px-3 py-2.5 text-left transition ${
                filter === key ? "border-[#117d70] bg-white shadow-sm" : "border-[#e9eeee] bg-white/60 hover:bg-white"
              }`}
            >
              <span className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#8b9697]">
                {Icon && <Icon size={12} />}
                {meta ? meta.label : "All parameters"}
              </span>
              <span className="mt-1 block text-xl font-bold tabular-nums text-[#263234]">{count}</span>
            </button>
          );
        })}
      </div>

      {report.aspects.map((aspect) => {
        const rows = aspect.rows.filter((row) => filter === "all" || row.status === filter);
        return (
          <section key={aspect.key} className="rounded-2xl border border-[#e9eeee] bg-white">
            <div className="border-b border-[#f0f3f3] px-4 py-3">
              <h3 className="text-[14px] font-semibold text-[#263234]">{aspect.title}</h3>
              {aspect.summary && <p className="mt-1 text-[12px] leading-5 text-[#6b7879]">{aspect.summary}</p>}
              {aspect.error && <p className="mt-1 text-[12px] text-red-600">{aspect.error}</p>}
            </div>
            {rows.length ? (
              <div className="scrollbar-subtle overflow-x-auto">
                <table className="w-full min-w-[640px] border-collapse text-left text-[12px]">
                  <thead>
                    <tr className="bg-[#fafbfb] text-[10px] uppercase tracking-[0.06em] text-[#8b9697]">
                      <th className="w-[18%] px-4 py-2 font-semibold">Parameter</th>
                      {report.documents.map((doc) => (
                        <th key={doc.document_id} className="px-3 py-2 font-semibold">
                          <span className="block max-w-[220px] truncate normal-case tracking-normal text-[11px] text-[#4e5a5c]">
                            {doc.filename}
                          </span>
                          {doc.document_id === report.baseline_document_id && (
                            <span className="mt-0.5 inline-flex items-center gap-1 text-[9px] text-[#087d70]">
                              <Scale size={9} /> Baseline
                            </span>
                          )}
                        </th>
                      ))}
                      <th className="w-[110px] px-3 py-2 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, index) => (
                      <tr key={`${row.parameter}-${index}`} className="border-t border-[#f0f3f3] align-top">
                        <td className="px-4 py-3 font-medium text-[#374345]">{row.parameter}</td>
                        {row.cells.map((cell) => (
                          <Cell key={cell.document_id} cell={cell} />
                        ))}
                        <td className="px-3 py-3">
                          <StatusBadge status={row.status} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              !aspect.error && (
                <p className="px-4 py-4 text-[12px] text-[#9aa5a5]">
                  {aspect.rows.length ? "No parameters with this status." : "No parameters found."}
                </p>
              )
            )}
          </section>
        );
      })}

      {report.missing_specifications.length > 0 && (filter === "all" || filter === "missing") && (
        <section className="rounded-2xl border border-[#e9eeee] bg-white p-4">
          <h3 className="flex items-center gap-2 text-[14px] font-semibold text-[#263234]">
            <CircleMinus size={15} className="text-[#a5650b]" />
            Missing specifications
          </h3>
          <ul className="mt-3 space-y-1.5 text-[12px] text-[#4e5a5c]">
            {report.missing_specifications.map((item, index) => (
              <li key={index} className="flex flex-wrap gap-x-1.5">
                <span className="font-medium">{item.aspect} – {item.parameter}:</span>
                <span className="text-[#8b9697]">not stated in {item.missing_in.join(", ")}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
      <p className="pb-4 text-center text-[10px] text-[#aab3b3]">
        Values are extracted by the model and each quote is checked against the PDF text. Values marked
        “unverified” could not be matched to the document; check them manually.
        {" "}Compared: {Object.values(names).join(" · ")}
      </p>
    </div>
  );
}

function StatusBadge({ status }) {
  const meta = STATUS[status];
  const Icon = meta.icon;
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-semibold ${meta.tone}`}>
      <Icon size={11} />
      {meta.label}
    </span>
  );
}

function Cell({ cell }) {
  const meta = ASSESSMENT[cell.assessment];
  if (cell.assessment === "not_specified") {
    return (
      <td className="px-3 py-3">
        <span className={`text-[11px] italic ${meta.tone}`}>Not specified</span>
      </td>
    );
  }
  return (
    <td className={`px-3 py-3 ${cell.assessment === "deviation" ? "bg-[#fef6f5]" : ""}`}>
      <div className="font-medium text-[#263234]">{cell.value}</div>
      <div className={`mt-0.5 flex flex-wrap items-center gap-x-1.5 text-[10px] ${meta.tone}`}>
        <span className="font-semibold">{meta.label}</span>
        {cell.page_number && <span className="text-[#9aa5a5]">· p.{cell.page_number}</span>}
        {!cell.verified && <span className="text-[#a5650b]">· unverified</span>}
      </div>
      {cell.note && <div className="mt-1 text-[11px] leading-4 text-[#b4372b]">{cell.note}</div>}
      {cell.quote && (
        <blockquote
          className="mt-1.5 line-clamp-3 border-l-2 border-[#dfe8e6] pl-2 text-[10.5px] italic leading-4 text-[#7a8788]"
          title={cell.quote}
        >
          “{cell.quote}”
        </blockquote>
      )}
    </td>
  );
}
