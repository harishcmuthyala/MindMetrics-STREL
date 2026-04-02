export default function LandingPage({ onStart }) {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-slate-100 px-4 py-16">
      <div className="w-full max-w-2xl rounded-[2rem] border border-slate-200 bg-white p-10 shadow-lg text-center">

        {/* University info */}
        <p className="text-xs font-semibold uppercase tracking-widest text-slate-400">
          University of Houston – Clear Lake
        </p>
        <p className="mt-1 text-xs text-slate-400">
          Spring 2026 &nbsp;·&nbsp; CSCI 6838 – Research Project
        </p>

        {/* Divider */}
        <div className="my-6 border-t border-slate-200" />

        {/* Title */}
        <h1 className="mt-4 text-4xl font-bold text-slate-800 leading-tight">
          MindMetrics
          <span className="block text-2xl font-semibold text-slate-500 mt-1">
            STREL Feature Analysis
          </span>
        </h1>

        {/* Tagline */}
        <p className="mt-2 text-sm font-medium text-cyan-600 uppercase tracking-widest">
          Stress &amp; Relaxation Detection
        </p>

        {/* Team */}
        <p className="mt-3 text-sm font-semibold text-emerald-600">
          Capstone Research Team
        </p>

        {/* Description */}
        <p className="mt-6 text-sm leading-7 text-slate-500 max-w-lg mx-auto">
          Explore which physiological and behavioral factors drive stress in
          individuals, and evaluate how machine learning models predict stress
          using the STREL dataset.
        </p>

        {/* CTA */}
        <button
          type="button"
          onClick={onStart}
          className="mt-10 inline-flex items-center justify-center rounded-2xl bg-emerald-500 px-8 py-3 text-sm font-semibold text-white transition hover:bg-emerald-600"
        >
          Get Started →
        </button>
      </div>
    </main>
  );
}
