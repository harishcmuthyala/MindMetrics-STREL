import { useEffect, useState } from "react";

const API_BASE_URL = "http://127.0.0.1:8000";

function formatTimestampForFilename(timestamp) {
  return timestamp.replace(/[:.]/g, "-");
}

function downloadJsonFile(filename, content) {
  const blob = new Blob([JSON.stringify(content, null, 2)], { type: "application/json" });
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(objectUrl);
}

export default function MainUI({ initialPresignedUrl = "", onBack }) {
  const [user, setUser] = useState("");
  const [notes, setNotes] = useState("");
  const [features, setFeatures] = useState([]);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [output, setOutput] = useState(null);
  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [runState, setRunState] = useState({ loading: false, error: "" });
  const [saveState, setSaveState] = useState({ local: "", s3: "", saving: false });

  useEffect(() => {
    async function loadFeatures() {
      try {
        setLoadingFeatures(true);
        const response = await fetch(`${API_BASE_URL}/features`);
        if (!response.ok) throw new Error("Could not load features");
        const data = await response.json();
        const incoming = data.features ?? [];
        setFeatures(incoming);
        setSelectedFeatures(incoming);
      } catch {
        setRunState({ loading: false, error: "Unable to load features from the backend." });
      } finally {
        setLoadingFeatures(false);
      }
    }
    loadFeatures();
  }, []);

  function toggleFeature(feature) {
    setSelectedFeatures((current) =>
      current.includes(feature)
        ? current.filter((item) => item !== feature)
        : [...current, feature]
    );
  }

  async function handleRun() {
    setRunState({ loading: true, error: "" });
    setSaveState({ local: "", s3: "", saving: false });

    try {
      const response = await fetch(`${API_BASE_URL}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_features: selectedFeatures, user, notes }),
      });
      if (!response.ok) throw new Error("Run failed");
      setOutput(await response.json());
    } catch {
      setRunState({ loading: false, error: "Run failed. Please check the backend and try again." });
      return;
    }

    setRunState({ loading: false, error: "" });
  }

  async function handleSave() {
    if (!output) return;

    const payloadToSave = {
      user,
      timestamp: new Date().toISOString(),
      notes,
      features_used: selectedFeatures,
      output,
    };
    const safeUser = user.trim() || "user";
    const filename = `${safeUser}_${formatTimestampForFilename(payloadToSave.timestamp)}.json`;

    setSaveState({ local: "", s3: "", saving: true });
    downloadJsonFile(filename, payloadToSave);
    const localStatus = "Saved locally";

    if (!initialPresignedUrl) {
      setSaveState({ local: localStatus, s3: "S3 failed. Get new URL from admin.", saving: false });
      return;
    }

    let s3Status = "";
    try {
      const response = await fetch(initialPresignedUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadToSave, null, 2),
      });
      s3Status = response.ok ? "Saved to S3" : "S3 failed. Get new URL from admin.";
    } catch {
      s3Status = "S3 failed. Get new URL from admin.";
    }

    setSaveState({ local: localStatus, s3: s3Status, saving: false });
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
                Choose features, add notes, run the model, then save the exact JSON payload locally and to S3.
              </p>
            </div>
            <div className="flex flex-col gap-3 md:items-end">
              <div className="rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                {selectedFeatures.length} of {features.length} features selected
              </div>
              <button
                type="button"
                onClick={onBack}
                className="inline-flex items-center justify-center rounded-2xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
              >
                Back to S3 URL
              </button>
            </div>
          </div>

          <div className="grid gap-6">

            {/* Section 1 - Name */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 1 - Team Member Name</h2>
              <input
                type="text"
                value={user}
                onChange={(e) => setUser(e.target.value)}
                placeholder="Your Name"
                className="mt-4 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
              />
            </section>

            {/* Section 2 - Feature Selection */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 2 - Feature Selection</h2>
              {loadingFeatures ? (
                <p className="mt-4 text-sm text-slate-400">Loading features...</p>
              ) : (
                <div className="mt-4 flex flex-wrap gap-3">
                  {features.map((feature) => {
                    const isSelected = selectedFeatures.includes(feature);
                    return (
                      <button
                        key={feature}
                        type="button"
                        onClick={() => toggleFeature(feature)}
                        className={`rounded-full border px-4 py-2 text-sm font-medium transition ${
                          isSelected
                            ? "border-emerald-500 bg-emerald-500 text-white"
                            : "border-slate-300 bg-white text-slate-600 hover:border-slate-400"
                        }`}
                      >
                        {feature}
                      </button>
                    );
                  })}
                </div>
              )}
            </section>

            {/* Section 3 - Notes */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 3 - Notes</h2>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                rows={5}
                placeholder="Notes / Reason for feature selection"
                className="mt-4 w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
              />
            </section>

            {/* Section 4 - Run */}
            <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
              <h2 className="text-lg font-semibold text-slate-800">Section 4 - Run</h2>
              <button
                type="button"
                onClick={handleRun}
                disabled={runState.loading || loadingFeatures}
                className="mt-4 inline-flex items-center justify-center rounded-2xl bg-cyan-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-cyan-600 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {runState.loading ? "Running..." : "Run Model"}
              </button>

              {runState.error && (
                <p className="mt-4 rounded-2xl border border-rose-300 bg-rose-50 px-4 py-3 text-sm text-rose-600">
                  {runState.error}
                </p>
              )}

              {output && (
                <pre className="mt-4 overflow-x-auto rounded-2xl border border-slate-200 bg-slate-100 px-4 py-4 text-sm leading-6 text-slate-700">
                  <code>{JSON.stringify(output, null, 2)}</code>
                </pre>
              )}
            </section>

            {/* Section 5 - Save */}
            {output && (
              <section className="rounded-3xl border border-slate-200 bg-slate-50 p-5">
                <h2 className="text-lg font-semibold text-slate-800">Section 5 - Save</h2>
                <button
                  type="button"
                  onClick={handleSave}
                  disabled={saveState.saving}
                  className="mt-4 inline-flex items-center justify-center rounded-2xl bg-amber-400 px-5 py-3 text-sm font-semibold text-slate-900 transition hover:bg-amber-500 disabled:cursor-not-allowed disabled:opacity-70"
                >
                  {saveState.saving ? "Saving..." : "Save"}
                </button>

                {(saveState.local || saveState.s3) && (
                  <div className="mt-4 space-y-2 text-sm">
                    {saveState.local && (
                      <p className="rounded-2xl border border-emerald-300 bg-emerald-50 px-4 py-3 text-emerald-700">
                        {saveState.local}
                      </p>
                    )}
                    {saveState.s3 && (
                      <p className={`rounded-2xl border px-4 py-3 ${
                        saveState.s3.startsWith("S3 failed")
                          ? "border-rose-300 bg-rose-50 text-rose-600"
                          : "border-sky-300 bg-sky-50 text-sky-700"
                      }`}>
                        {saveState.s3}
                      </p>
                    )}
                  </div>
                )}
              </section>
            )}

          </div>
        </section>
      </div>
    </main>
  );
}
