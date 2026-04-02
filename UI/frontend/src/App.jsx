import { useState } from "react";
import MainUI from "./MainUI";
import LandingPage from "./LandingPage";

const verificationPayload = JSON.stringify({ verification: true });

export default function App() {
  const [screen, setScreen] = useState("landing");
  const [presignedUrl, setPresignedUrl] = useState("");
  const [draftUrl, setDraftUrl] = useState("");
  const [status, setStatus] = useState({ type: "idle", message: "" });
  const [isVerifying, setIsVerifying] = useState(false);
  const [skipVerification, setSkipVerification] = useState(false);

  async function verifyUrl(event) {
    event.preventDefault();
    setStatus({ type: "idle", message: "" });
    setIsVerifying(true);

    try {
      const response = await fetch(draftUrl, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: verificationPayload,
      });

      if (!response.ok) throw new Error("Verification failed");

      setPresignedUrl(draftUrl);
      setStatus({ type: "success", message: "URL verified. Loading app..." });
    } catch (error) {
      setStatus({ type: "error", message: "Invalid or expired URL. Get a new one from admin." });
    } finally {
      setIsVerifying(false);
    }
  }

  function handleSkip() {
    setPresignedUrl(draftUrl);
    setSkipVerification(true);
    setStatus({ type: "warning", message: "Validation was skipped. Saving to S3 may fail if the URL is invalid or expired." });
  }

  function handleBackToVerification() {
    setDraftUrl(presignedUrl);
    setPresignedUrl("");
    setSkipVerification(false);
    setStatus({ type: "idle", message: "" });
  }

  if (presignedUrl || skipVerification) {
    return <MainUI initialPresignedUrl={presignedUrl} onBack={handleBackToVerification} />;
  }

  if (screen === "landing") {
    return <LandingPage onStart={() => setScreen("verify")} />;
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-12 bg-slate-100">
      <section className="w-full max-w-xl rounded-3xl border border-slate-200 bg-white p-8 shadow-lg">
        <div className="mb-8">
          <p className="text-sm font-semibold uppercase tracking-[0.3em] text-emerald-600">
            Screen 1
          </p>
          <h1 className="mt-3 text-4xl font-semibold text-slate-800">
            S3 URL Verification
          </h1>
          <p className="mt-3 text-sm leading-6 text-slate-500">
            Paste the presigned S3 upload URL provided by your admin and verify
            it before continuing. Validating now is strongly recommended because
            save-to-S3 depends on this URL being active.
          </p>
        </div>

        <div className="mb-6 rounded-2xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-700">
          Verifying now is important. Skip only if you need to continue
          temporarily and understand S3 save may fail later.
        </div>

        <form className="space-y-5" onSubmit={verifyUrl}>
          <label className="block">
            <span className="mb-2 block text-sm font-medium text-slate-700">
              Paste S3 Presigned URL
            </span>
            <input
              type="url"
              required
              value={draftUrl}
              onChange={(e) => setDraftUrl(e.target.value)}
              placeholder="https://bucket.s3.amazonaws.com/..."
              className="w-full rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm text-slate-800 outline-none transition focus:border-emerald-500 focus:ring-2 focus:ring-emerald-100"
            />
          </label>

          <div className="flex flex-col gap-3 sm:flex-row">
            <button
              type="submit"
              disabled={isVerifying}
              className="inline-flex flex-1 items-center justify-center rounded-2xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {isVerifying ? "Verifying..." : "Verify"}
            </button>
            <button
              type="button"
              onClick={handleSkip}
              disabled={isVerifying || !draftUrl}
              className="inline-flex flex-1 items-center justify-center rounded-2xl border border-slate-300 bg-white px-4 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-70"
            >
              Skip for now
            </button>
          </div>
        </form>

        {status.message && (
          <p className={`mt-5 rounded-2xl border px-4 py-3 text-sm ${
            status.type === "error"
              ? "border-rose-300 bg-rose-50 text-rose-600"
              : status.type === "warning"
                ? "border-amber-300 bg-amber-50 text-amber-700"
                : "border-emerald-300 bg-emerald-50 text-emerald-700"
          }`}>
            {status.message}
          </p>
        )}
      </section>
    </main>
  );
}
