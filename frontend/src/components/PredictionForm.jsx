import React, { useState } from 'react';
import { Sparkles, Cpu, ShieldCheck, AlertCircle, ArrowUpRight } from 'lucide-react';

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
    label: "Executive Liquidation",
    ticker: "XYZ",
    title: "Chief Executive Officer liquidates 25% of personal equity in open market transactions",
    description: "SEC Form 4 filings reveal unplanned discretionary stock disposition totaling $45 million."
  },
  {
    label: "Revenue Surge",
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
    <div className="bg-[#242720] border border-[#393C33] rounded p-6 shadow-sm">
      {/* Section Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between pb-4 mb-5 border-b border-[#393C33] gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-[#A18B62] uppercase tracking-wider">
              CATALYST INGESTION ENGINE
            </span>
          </div>
          <h2 className="text-base font-semibold text-[#E8E4D9] mt-0.5 font-mono">
            Evaluate Corporate News Event
          </h2>
          <p className="text-xs text-[#85877C] mt-0.5">
            Submit a real-world news headline or filing context to forecast FINRA short-interest trajectory.
          </p>
        </div>

        {/* Sample Scenario Buttons */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-[11px] text-[#85877C] font-mono mr-1">SCENARIOS:</span>
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleApplyPreset(p)}
              className="text-[11px] font-mono px-2.5 py-1 rounded border border-[#393C33] bg-[#1D201B] hover:bg-[#2B2E27] text-[#AAA99D] hover:text-[#E8E4D9] hover:border-[#A18B62]/60 transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 rounded border border-[#B66F58]/40 bg-[#B66F58]/10 flex items-start gap-2.5 text-[#B66F58] text-xs">
          <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
          <div>
            <strong className="font-mono uppercase font-semibold">Evaluation Error: </strong>
            <span>{error}</span>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Row 1: Headline & Ticker */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-3">
            <label htmlFor="headline-input" className="block text-[11px] font-mono uppercase text-[#AAA99D] mb-1.5">
              Event Title / Headline <span className="text-[#B66F58]">*</span>
            </label>
            <input
              id="headline-input"
              type="text"
              required
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Major Automaker recalls 500,000 electric vehicles citing battery defect"
              className="w-full px-3.5 py-2.5 bg-[#1D201B] border border-[#393C33] rounded text-sm text-[#E8E4D9] placeholder-[#85877C] focus:outline-none focus:border-[#A18B62] transition-colors"
            />
          </div>

          <div>
            <label htmlFor="ticker-input" className="block text-[11px] font-mono uppercase text-[#AAA99D] mb-1.5">
              Ticker Symbol <span className="text-[#85877C]">(OPTIONAL)</span>
            </label>
            <input
              id="ticker-input"
              type="text"
              maxLength={10}
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. TSLA"
              className="w-full px-3.5 py-2.5 bg-[#1D201B] border border-[#393C33] rounded text-sm text-[#E8E4D9] placeholder-[#85877C] focus:outline-none focus:border-[#A18B62] transition-colors font-mono uppercase"
            />
          </div>
        </div>

        {/* Row 2: Description */}
        <div>
          <label htmlFor="description-input" className="block text-[11px] font-mono uppercase text-[#AAA99D] mb-1.5">
            Detailed News Description / SEC Filing Context <span className="text-[#85877C]">(OPTIONAL)</span>
          </label>
          <textarea
            id="description-input"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Paste additional context, article excerpts, executive statements, or Form 4 transaction remarks..."
            className="w-full px-3.5 py-2 bg-[#1D201B] border border-[#393C33] rounded text-xs text-[#E8E4D9] placeholder-[#85877C] focus:outline-none focus:border-[#A18B62] transition-colors resize-none leading-relaxed"
          />
        </div>

        {/* Row 3: Model Family & Gemini Toggle */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
          {/* Model Selection */}
          <div className="p-3 bg-[#1D201B] border border-[#393C33] rounded">
            <span className="block text-[11px] font-mono uppercase text-[#AAA99D] mb-2">
              Select ML Model Family
            </span>
            <div className="grid grid-cols-2 gap-2">
              <label className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
                modelFamily === 'logistic_regression'
                  ? 'bg-[#2B2E27] border-[#A18B62] text-[#E8E4D9]'
                  : 'bg-[#242720] border-[#393C33] text-[#85877C] hover:text-[#AAA99D]'
              }`}>
                <input
                  type="radio"
                  name="model_family"
                  value="logistic_regression"
                  checked={modelFamily === 'logistic_regression'}
                  onChange={() => setModelFamily('logistic_regression')}
                  className="sr-only"
                />
                <Cpu className={`h-4 w-4 shrink-0 ${modelFamily === 'logistic_regression' ? 'text-[#A18B62]' : 'text-[#85877C]'}`} />
                <div>
                  <div className="text-xs font-mono font-medium">LOGISTIC REGRESSION</div>
                  <div className="text-[10px] text-[#85877C] font-mono">30k TF-IDF + Struct</div>
                </div>
              </label>

              <label className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors ${
                modelFamily === 'xgboost'
                  ? 'bg-[#2B2E27] border-[#A18B62] text-[#E8E4D9]'
                  : 'bg-[#242720] border-[#393C33] text-[#85877C] hover:text-[#AAA99D]'
              }`}>
                <input
                  type="radio"
                  name="model_family"
                  value="xgboost"
                  checked={modelFamily === 'xgboost'}
                  onChange={() => setModelFamily('xgboost')}
                  className="sr-only"
                />
                <ShieldCheck className={`h-4 w-4 shrink-0 ${modelFamily === 'xgboost' ? 'text-[#A18B62]' : 'text-[#85877C]'}`} />
                <div>
                  <div className="text-xs font-mono font-medium">XGBOOST</div>
                  <div className="text-[10px] text-[#85877C] font-mono">Gradient Boosted Trees</div>
                </div>
              </label>
            </div>
          </div>

          {/* Gemini Enrichment Toggle */}
          <div className="p-3 bg-[#1D201B] border border-[#393C33] rounded flex flex-col justify-between">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-[11px] font-mono uppercase text-[#AAA99D] flex items-center gap-1.5">
                  <Sparkles className="h-3.5 w-3.5 text-[#A18B62]" />
                  Gemini LLM Enrichment
                </span>
                <p className="text-[11px] text-[#85877C] mt-0.5">
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
                <div className="w-8 h-4 bg-[#242720] border border-[#393C33] rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-[#171916] after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-[#85877C] peer-checked:after:bg-[#171916] after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-[#A18B62]"></div>
              </label>
            </div>
            <div className="text-[10px] font-mono text-[#85877C] mt-2">
              {useLlm 
                ? "STATUS: Grounded Gemini 3.5 API with anti-fabrication verbatim check." 
                : "STATUS: Deterministic regular expression keyword heuristics."}
            </div>
          </div>
        </div>

        {/* Generate Prediction Button */}
        <div className="pt-2 flex items-center justify-end">
          <button
            type="submit"
            disabled={isLoading || !title.trim()}
            className="w-full sm:w-auto px-6 py-2.5 rounded font-mono font-semibold text-xs tracking-wider uppercase text-[#171916] bg-[#A18B62] hover:bg-[#B29B70] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-3.5 w-3.5 text-[#171916]" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                </svg>
                PROCESSING CATALYST...
              </>
            ) : (
              <>
                <span>Generate Prediction</span>
                <ArrowUpRight className="h-3.5 w-3.5" />
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
