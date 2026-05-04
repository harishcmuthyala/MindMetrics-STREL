import { useEffect, useRef, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";
const PLOTS = [
  { key: "confusion_matrices.png", label: "Confusion Matrices" },
  { key: "model_comparison.png",   label: "Model Comparison" },
  { key: "roc_curves.png",         label: "ROC Curves" },
  { key: "fold_accuracy.png",      label: "Fold Accuracy" },
];

export default function MainUI({ initialPresignedUrl = "", onBack }) {
  const [user, setUser] = useState("");
  const [notes, setNotes] = useState("");
  const [features, setFeatures] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [output, setOutput] = useState(null);
  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [runState, setRunState] = useState({ loading: false, error: "" });
  const [saveState, setSaveState] = useState({ local: "", s3: "", s3Failed: false });
  const [elapsed, setElapsed] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);
  const timerRef = useRef(null);
  const startTimeRef = useRef(null);

  useEffect(() => {
    async function loadFeatures() {
      try {
        const response = await fetch(`${API_BASE_URL}/features`);
        if (!response.ok) throw new Error();
        const data = await response.json();
        const categories = data.categories ?? [];
        setFeatures(categories);
        setSelectedFeatures(categories.flatMap((c) => c.features));
      } catch {
        setRunState({ loading: false, error: "Unable to load features from the backend." });
      } finally {
        setLoadingFeatures(false);
      }
    }
    loadFeatures();
  }, []);

  function toggleFeature(feature) {
    setSelectedFeatures((cur) =>
      cur.includes(feature) ? cur.filter((f) => f !== feature) : [...cur, feature]
    );
  }

  function startTimer() {
    setElapsed(0);
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  }

  function stopTimer() { clearInterval(timerRef.current); }

  function formatElapsed(s) {
    return `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;
  }

  async function attemptS3Save(payload) {
    if (!initialPresignedUrl) {
      setSaveState((s) => ({ ...s, s3: "No S3 URL provided.", s3Failed: true }));
      return;
    }
    try {
      const res = await fetch(initialPresignedUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload, null, 2),
      });
      if (!res.ok) throw new Error();
      setSaveState((s) => ({ ...s, s3: "Saved to S3", s3Failed: false }));
    } catch {
      setSaveState((s) => ({ ...s, s3: "S3 save failed.", s3Failed: true }));
    }
  }

  async function autoSave(result) {
    const timestamp = new Date().toISOString();
    const allFeatures = features.flatMap((c) => c.features);
    const excludedFeatures = allFeatures.filter((f) => !selectedFeatures.includes(f));
    const payload = { user, timestamp, notes, features_used: selectedFeatures, features_excluded: excludedFeatures, output: result };

    try {
      await fetch(`${API_BASE_URL}/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload, selected_features: selectedFeatures, excluded_features: excludedFeatures, output: result }),
      });
      setSaveState((s) => ({ ...s, local: "Saved to results/runs/" }));
    } catch {
      setSaveState((s) => ({ ...s, local: "Local save failed." }));
    }

    await attemptS3Save(payload);
  }

  async function handleRun() {
    setRunState({ loading: true, error: "" });
    setSaveState({ local: "", s3: "", s3Failed: false });
    setOutput(null);
    setSelectedModel(null);
    startTimer();

    let result;
    try {
      const response = await fetch(`${API_BASE_URL}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_features: selectedFeatures, user, notes }),
      });
      if (!response.ok) throw new Error();
      result = await response.json();
    } catch {
      setRunState({ loading: false, error: "Run failed. Please check the backend and try again." });
      stopTimer();
      return;
    }

    stopTimer();
    setRunState({ loading: false, error: "" });
    setOutput(result);
    autoSave(result);
  }

  return (
    <main className="min-h-screen bg-slate-100 px-4 py-10">
      <div className="mx-auto max-w-5xl">
        <section className="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-lg md:p-8">

          {/* Header */}
          <div className="mb-8 flex flex-col gap-3 border-b border-slate-200 pb-6 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.3em] text-emerald-600">Screen 2</p>
              <h1 className="mt-3 text-4xl font-semibold text-slate-800">Feature Runner</h1>
              <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
                Choose features, add notes, then run. Results are auto-saved to the repo and S3.
              </p>
            </div>
            <div className="flex flex-col gap-3 md:items-end">
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                {selectedFeatures.length} of {features.flatMap((c) => c.features).length} features selected
              </div>
              <button type="button" onClick={onBack}
                className="inline-flex items-center justify-center rounded-2xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">
                Back to S3 URL
              </button>
            </div>
          </div>

          <div className="grid gap-6">

            {/* Section 1 - Name */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 1 - Team Member Name</h2>
              <input type="text" value={user} onChange={(e) => setUser(e.target.value)}
                placeholder="Your Name"
                className="mt-4 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100" />
            </section>

            {/* Section 2 - Feature Selection */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 2 - Feature Selection</h2>
              {loadingFeatures ? (
                <p className="mt-4 text-sm text-slate-400">Loading features...</p>
              ) : (
                <div className="mt-4 space-y-5">
                  {features.map((category) => (
                    <div key={category.label}>
                      <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">{category.label}</p>
                      <div className="flex flex-wrap gap-2">
                        {category.features.map((feature, idx) => {
                          const isSelected = selectedFeatures.includes(feature);
                          return (
                            <button key={feature} type="button"
                              title={category.tooltips?.[idx] ?? ""}
                              onClick={() => toggleFeature(feature)}
                              className={`rounded-full border px-4 py-2 text-sm font-medium transition ${
                                isSelected
                                  ? "border-emerald-500 bg-emerald-500 text-white"
                                  : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
                              }`}>
                              {feature}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </section>

            {/* Section 3 - Notes */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 3 - Notes</h2>
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={5}
                placeholder="Notes / Reason for feature selection"
                className="mt-4 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100" />
            </section>

            {/* Section 4 - Run */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 4 - Run</h2>
              <button type="button" onClick={handleRun}
                disabled={runState.loading || loadingFeatures}
                className="mt-4 inline-flex items-center justify-center rounded-2xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-70">
                {runState.loading ? `Running... ${formatElapsed(elapsed)}` : "Run Model"}
              </button>

              {runState.error && (
                <p className="mt-4 rounded-2xl border border-rose-300 bg-rose-50 px-4 py-3 text-sm text-rose-600">{runState.error}</p>
              )}

              {/* Save status — shown after run */}
              {(saveState.local || saveState.s3) && (
                <div className="mt-4 space-y-2">
                  {saveState.local && (
                    <p className="rounded-2xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                      {saveState.local}
                    </p>
                  )}
                  {saveState.s3 && (
                    <div className="flex items-center gap-3">
                      <p className={`flex-1 rounded-2xl border px-4 py-3 text-sm ${
                        saveState.s3Failed
                          ? "border-rose-300 bg-rose-50 text-rose-600"
                          : "border-sky-300 bg-sky-50 text-sky-700"
                      }`}>
                        {saveState.s3}
                      </p>
                      {saveState.s3Failed && (
                        <button type="button" onClick={() => attemptS3Save({ user, timestamp: new Date().toISOString(), notes, features_used: selectedFeatures, output })}
                          className="rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50">
                          Retry S3
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}

              {output && (
                <div className="mt-6 space-y-6">

                  {/* Model Summary */}
                  <div>
                    <p className="mb-2 text-sm font-semibold text-slate-700">Model Summary — click a row to see fold breakdown</p>
                    <div className="overflow-x-auto rounded-2xl border border-slate-200">
                      <table className="w-full text-sm text-slate-700">
                        <thead className="bg-slate-100 text-xs font-semibold uppercase tracking-wider text-slate-500">
                          <tr>
                            {["Model", "Accuracy", "Precision", "Recall", "F1-Score", "TN", "FP", "FN", "TP"].map((h) => (
                              <th key={h} className="px-4 py-3 text-left">{h}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {output.model_metrics.map((row, i) => {
                            const isActive = selectedModel === row.Model;
                            return (
                              <tr key={i} onClick={() => setSelectedModel(isActive ? null : row.Model)}
                                className={`cursor-pointer transition ${isActive ? "bg-emerald-50" : "hover:bg-slate-50"}`}>
                                <td className="px-4 py-3 font-medium">
                                  {row.Model}{isActive && <span className="ml-2 text-xs text-emerald-600">▾</span>}
                                </td>
                                {["Accuracy", "Precision", "Recall", "F1-Score"].map((k) => (
                                  <td key={k} className="px-4 py-3">{(row[k] * 100).toFixed(1)}%</td>
                                ))}
                                {["TN", "FP", "FN", "TP"].map((k) => (
                                  <td key={k} className="px-4 py-3">{row[k]}</td>
                                ))}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Fold Breakdown */}
                  {selectedModel && (
                    <div>
                      <p className="mb-2 text-sm font-semibold text-slate-700">Fold Breakdown — {selectedModel}</p>
                      <div className="overflow-x-auto rounded-2xl border border-slate-200">
                        <table className="w-full text-sm text-slate-700">
                          <thead className="bg-slate-100 text-xs font-semibold uppercase tracking-wider text-slate-500">
                            <tr>
                              {["Fold", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"].map((h) => (
                                <th key={h} className="px-4 py-3 text-left">{h}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {output.fold_metrics.filter((r) => r.model === selectedModel).map((row, i) => (
                              <tr key={i} className="hover:bg-slate-50">
                                <td className="px-4 py-3 font-medium">Fold {row.fold}</td>
                                {["accuracy", "precision", "recall", "f1_score", "roc_auc"].map((k) => (
                                  <td key={k} className="px-4 py-3">{(row[k] * 100).toFixed(1)}%</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Plots */}
                  <div>
                    <p className="mb-3 text-sm font-semibold text-slate-700">Result Plots</p>
                    <div className="grid gap-6 sm:grid-cols-2">
                      {PLOTS.map(({ key, label }) => (
                        <div key={key} className="rounded-2xl border border-slate-200 bg-white p-3">
                          <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-slate-400">{label}</p>
                          <img src={`${API_BASE_URL}/results/${key}?t=${Date.now()}`} alt={label} className="w-full rounded-xl" />
                        </div>
                      ))}
                    </div>
                  </div>

                </div>
              )}
            </section>

          </div>
        </section>
      </div>
    </main>
  );
}
