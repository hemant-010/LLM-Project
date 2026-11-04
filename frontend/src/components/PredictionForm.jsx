import React, { useState } from 'react';
import { Send, Sparkles, Cpu, ShieldCheck, Zap, AlertCircle } from 'lucide-react';

const PRESETS = [
  {
    label: "Vehicle Recall",
    ticker: "TSLA",
    title: "Tesla recalls 1.5 million vehicles in US due to power steering software glitch",
    description: "National Highway Traffic Safety Administration issues formal safety defect notice affecting Model 3 and Model Y."
  },
  {
    label: "Aircraft Grounding",
    ticker: "BA",
    title: "Boeing 737 Max 9 aircraft grounded by FAA following mid-air fuselage breach",
    description: "Federal Aviation Administration orders immediate emergency airworthiness directives and safety audits."
  },
  {
    label: "Executive Sale",
    ticker: "XYZ",
    title: "Chief Executive Officer liquidates 25% of personal equity in open market transactions",
    description: "SEC Form 4 filings reveal unplanned discretionary stock disposition totaling $45 million."
  },
  {
    label: "Positive Earnings",
    ticker: "NVDA",
    title: "NVIDIA reports 206% year-over-year revenue surge driven by data center AI demand",
    description: "Quarterly earnings beat consensus Wall Street projections with forward guidance raised substantially."
  }
];

export default function PredictionForm({ onSubmit, isLoading, error }) {
  const [title, setTitle] = useState('');
  const [ticker, setTicker] = useState('');
  const [description, setDescription] = useState('');
  const [modelFamily, setModelFamily] = useState('logistic_regression');
  const [useLlm, setUseLlm] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim()) return;

    onSubmit({
      title: title.trim(),
      ticker: ticker.trim().toUpperCase() || undefined,
      description: description.trim() || undefined,
      model_family: modelFamily,
      use_llm: useLlm
    });
  };

  const handleApplyPreset = (preset) => {
    setTitle(preset.title);
    setTicker(preset.ticker);
    setDescription(preset.description);
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg">
      <div className="flex items-center justify-between pb-3 mb-4 border-b border-slate-800">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Zap className="h-4 w-4 text-blue-400" />
            Evaluate Corporate News Event
          </h2>
          <p className="text-xs text-slate-400">
            Submit a real-world news headline or filing context to forecast FINRA short-interest trajectory.
          </p>
        </div>

        {/* Quick Presets */}
        <div className="hidden lg:flex items-center gap-1.5">
          <span className="text-[11px] text-slate-500 font-medium mr-1">Sample Scenarios:</span>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleApplyPreset(p)}
              className="text-[11px] px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors border border-slate-700/60"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded-lg bg-rose-950/40 border border-rose-500/40 flex items-start gap-2.5 text-rose-300 text-xs">
          <AlertCircle className="h-4 w-4 text-rose-400 mt-0.5 shrink-0" />
          <div>
            <strong className="font-semibold">Prediction Error: </strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Row 1: Headline & Ticker */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-3">
            <label htmlFor="headline-input" className="block text-xs font-semibold uppercase text-slate-300 mb-1">
              Event Title / Headline <span className="text-rose-400">*</span>
            </label>
            <input
              id="headline-input"
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Major Automaker recalls 500,000 electric vehicles citing battery defect"
              className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-sans"
            />
          </div>

          <div>
            <label htmlFor="ticker-input" className="block text-xs font-semibold uppercase text-slate-300 mb-1">
              Ticker Symbol <span className="text-slate-500 font-normal">(Optional)</span>
            </label>
            <input
              id="ticker-input"
              type="text"
              maxLength={10}
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. TSLA"
              className="w-full px-3.5 py-2.5 bg-slate-950 border border-slate-700 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono uppercase"
            />
          </div>
        </div>

        {/* Row 2: Description */}
        <div>
          <label htmlFor="description-input" className="block text-xs font-semibold uppercase text-slate-300 mb-1">
            Detailed News Description / SEC Filing Extract <span className="text-slate-500 font-normal">(Optional context)</span>
          </label>
          <textarea
            id="description-input"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Paste additional context, article excerpts, executive statements, or Form 4 transaction remarks..."
            className="w-full px-3.5 py-2 bg-slate-950 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-sans resize-none"
          />
        </div>

        {/* Row 3: Model Family & Gemini Toggle */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          {/* Model Selection */}
          <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg">
            <span className="block text-xs font-semibold uppercase text-slate-300 mb-2">
              Select ML Model Family
            </span>
            <div className="grid grid-cols-2 gap-2">
              <label className={`flex items-center gap-2 p-2 rounded-md border cursor-pointer transition-all ${
                modelFamily === 'logistic_regression'
                  ? 'bg-blue-950/30 border-blue-500/50 text-blue-300 ring-1 ring-blue-500/30'
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}>
                <input
                  type="radio"
                  name="model_family"
                  value="logistic_regression"
                  checked={modelFamily === 'logistic_regression'}
                  onChange={() => setModelFamily('logistic_regression')}
                  className="sr-only"
                />
                <Cpu className="h-4 w-4 text-blue-400 shrink-0" />
                <div>
                  <div className="text-xs font-semibold">Logistic Regression</div>
                  <div className="text-[10px] text-slate-500">TF-IDF (30k) + Struct</div>
                </div>
              </label>

              <label className={`flex items-center gap-2 p-2 rounded-md border cursor-pointer transition-all ${
                modelFamily === 'xgboost'
                  ? 'bg-indigo-950/30 border-indigo-500/50 text-indigo-300 ring-1 ring-indigo-500/30'
                  : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700'
              }`}>
                <input
                  type="radio"
                  name="model_family"
                  value="xgboost"
                  checked={modelFamily === 'xgboost'}
                  onChange={() => setModelFamily('xgboost')}
                  className="sr-only"
                />
                <ShieldCheck className="h-4 w-4 text-indigo-400 shrink-0" />
                <div>
                  <div className="text-xs font-semibold">XGBoost</div>
                  <div className="text-[10px] text-slate-500">Gradient Boosted Trees</div>
                </div>
              </label>
            </div>
          </div>

          {/* Gemini Enrichment Toggle */}
          <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-semibold uppercase text-slate-300 flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-cyan-400" />
                  Gemini LLM Enrichment
                </span>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Strict verbatim evidence extraction & contextual risk analysis.
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer ml-2">
                <input
                  type="checkbox"
                  checked={useLlm}
                  onChange={(e) => setUseLlm(e.target.checked)}
                  className="sr-only peer"
                />
                <div className="w-9 h-5 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-cyan-600"></div>
              </label>
            </div>
            <div className="text-[10px] text-slate-500 mt-2">
              {useLlm 
                ? "Active: Invoking Gemini API with strict anti-fabrication substring checks." 
                : "Inactive: Utilizing fast deterministic regex keyword heuristics."}
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="pt-2 flex items-center justify-end">
          <button
            type="submit"
            disabled={isLoading || !title.trim()}
            className="w-full sm:w-auto px-6 py-2.5 rounded-lg font-semibold text-xs text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md shadow-blue-600/20 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-4 w-4 text-white" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                </svg>
                Evaluating Event & Forecasting Short-Interest...
              </>
            ) : (
              <>
                <Send className="h-3.5 w-3.5" />
                Generate Short-Interest Prediction
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

