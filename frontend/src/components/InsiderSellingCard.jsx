import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  Bell, 
  Users, 
  Clock, 
  ArrowRight, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp, 
  TrendingDown, 
  CheckCircle2, 
  AlertTriangle, 
  Search, 
  FileText,
  Sparkles,
  RefreshCw
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';
const COVERAGE_TICKERS = ['TSLA', 'AAPL', 'AMZN', 'NVDA', 'GM'];

export default function InsiderSellingCard({ onEvaluateTransaction }) {
  const [selectedTicker, setSelectedTicker] = useState('TSLA');
  const [customTicker, setCustomTicker] = useState('');
  const [transactions, setTransactions] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('screener'); // 'screener' | 'alerts'
  const [isLoading, setIsLoading] = useState(false);
  const [expandedId, setExpandedId] = useState(null);
  const [error, setError] = useState(null);

  // Fetch transactions for selected ticker
  const fetchTransactions = async (ticker) => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/insider-transactions?ticker=${ticker}&limit=10`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to load filings`);
      const data = await res.json();
      setTransactions(data.transactions || []);
    } catch (err) {
      console.error('Failed to load insider transactions:', err);
      setError('Unable to load insider transactions. Check backend connection.');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch cross-ticker alert feed
  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/insider-alerts`);
      if (res.ok) {
        const data = await res.json();
        setAlerts(data.alerts || []);
      }
    } catch (err) {
      console.error('Failed to load insider alerts:', err);
    }
  };

  useEffect(() => {
    fetchTransactions(selectedTicker);
    fetchAlerts();
  }, [selectedTicker]);

  const handleTickerSelect = (sym) => {
    setSelectedTicker(sym);
    setCustomTicker('');
  };

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    const sym = customTicker.trim().toUpperCase();
    if (sym) {
      setSelectedTicker(sym);
    }
  };

  const toggleExpand = (idx) => {
    setExpandedId(expandedId === idx ? null : idx);
  };

  const formatCurrency = (val) => {
    if (val == null) return 'N/A';
    if (val >= 1e9) return `$${(val / 1e9).toFixed(2)}B`;
    if (val >= 1e6) return `$${(val / 1e6).toFixed(2)}M`;
    if (val >= 1e3) return `$${(val / 1e3).toFixed(1)}k`;
    return `$${val.toLocaleString()}`;
  };

  const handleForecastClick = (tx) => {
    if (!onEvaluateTransaction) return;

    const valFormatted = formatCurrency(tx.estimated_value);
    const planText = tx.is_10b5_1 ? 'under pre-scheduled Rule 10b5-1 plan' : 'in open-market discretionary sale';
    const headline = `${tx.ticker}: ${tx.insider_name} (${tx.insider_role}) sells ${tx.shares_sold?.toLocaleString()} shares (${valFormatted}) ${planText}`;
    
    const holdingText = tx.percentage_holdings_sold != null 
      ? `${tx.percentage_holdings_sold.toFixed(1)}% of prior holdings` 
      : 'holding denominator unverified';
      
    const desc = `SEC Form 4 filing date: ${tx.filing_date || 'N/A'}, transaction date: ${tx.transaction_date || 'N/A'}. Liquidated: ${holdingText}. Evaluated risk score: ${tx.risk_score?.toFixed(2) || 'N/A'} (${tx.significance_category}).`;

    onEvaluateTransaction({
      title: headline,
      ticker: tx.ticker,
      description: desc,
      model_family: 'logistic_regression',
      use_llm: false
    });
  };

  const displayedList = activeTab === 'screener' ? transactions : alerts;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 border-b border-slate-800 gap-2">
        <div className="flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-amber-400" />
          <h3 className="text-base font-bold text-white tracking-wide">
            Executive Insider Selling & SEC Form 4 Intelligence
          </h3>
          <span className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            Active Module
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('screener')}
            className={`px-3 py-1 text-xs font-medium rounded-lg transition-colors ${
              activeTab === 'screener'
                ? 'bg-blue-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 bg-slate-800/60'
            }`}
          >
            Ticker Screener
          </button>
          <button
            onClick={() => setActiveTab('alerts')}
            className={`px-3 py-1 text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors ${
              activeTab === 'alerts'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200 bg-slate-800/60'
            }`}
          >
            <Bell className="h-3 w-3" />
            Alert Feed ({alerts.length})
          </button>
        </div>
      </div>

      {/* Subheader info & Ticker Navigation */}
      {activeTab === 'screener' && (
        <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-slate-400 font-medium mr-1">Institutional Coverage:</span>
            {COVERAGE_TICKERS.map((sym) => (
              <button
                key={sym}
                onClick={() => handleTickerSelect(sym)}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                  selectedTicker === sym
                    ? 'bg-blue-500 text-white font-bold shadow-sm'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                ${sym}
              </button>
            ))}
          </div>

          <form onSubmit={handleCustomSubmit} className="flex items-center gap-1.5">
            <input
              type="text"
              placeholder="Look up ticker..."
              value={customTicker}
              onChange={(e) => setCustomTicker(e.target.value)}
              className="px-2.5 py-1 text-xs bg-slate-900 border border-slate-700 rounded text-slate-200 uppercase font-mono w-28 focus:outline-none focus:border-blue-500"
            />
            <button
              type="submit"
              className="px-2.5 py-1 text-xs bg-slate-800 text-slate-300 hover:text-white hover:bg-slate-700 rounded flex items-center gap-1"
            >
              <Search className="h-3 w-3" />
              Scan
            </button>
          </form>
        </div>
      )}

      {/* Loading & Error States */}
      {isLoading && (
        <div className="py-8 flex flex-col items-center justify-center text-slate-400 text-xs">
          <RefreshCw className="h-5 w-5 animate-spin text-blue-400 mb-2" />
          Fetching verified SEC Form 4 filings...
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-950/40 border border-red-800/60 rounded-lg text-xs text-red-300 flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Filings / Alerts List */}
      {!isLoading && !error && displayedList.length === 0 && (
        <div className="py-6 text-center text-slate-500 text-xs bg-slate-950/40 rounded-lg border border-slate-800/60">
          No significant insider transactions found matching criteria.
        </div>
      )}

      {!isLoading && !error && displayedList.length > 0 && (
        <div className="space-y-3">
          {displayedList.map((tx, idx) => {
            const isExpanded = expandedId === idx;
            const isSignificant = tx.significance_category === 'significant';
            const isReview = tx.significance_category === 'requires_review';

            return (
              <div
                key={idx}
                className={`rounded-lg border transition-all ${
                  isSignificant
                    ? 'bg-red-950/20 border-red-800/60 shadow-sm'
                    : isReview
                    ? 'bg-amber-950/20 border-amber-800/50'
                    : 'bg-slate-950/60 border-slate-800/80 hover:border-slate-700'
                }`}
              >
                <div className="p-3.5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    {/* Insider info */}
                    <div className="flex items-start gap-2.5">
                      <div className="mt-0.5">
                        {isSignificant ? (
                          <ShieldAlert className="h-4 w-4 text-red-400 shrink-0" />
                        ) : isReview ? (
                          <AlertTriangle className="h-4 w-4 text-amber-400 shrink-0" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4 text-slate-400 shrink-0" />
                        )}
                      </div>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-bold text-white text-sm">{tx.insider_name}</span>
                          <span className="text-xs text-slate-400">({tx.insider_role})</span>
                          {tx.is_c_suite && (
                            <span className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30">
                              C-Suite Officer
                            </span>
                          )}
                          <span className="font-mono text-xs font-semibold text-blue-400">${tx.ticker}</span>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] text-slate-500 mt-1">
                          <span>Filing: {tx.filing_date || 'N/A'}</span>
                          <span>&bull;</span>
                          <span>Tx: {tx.transaction_date || 'N/A'}</span>
                        </div>
                      </div>
                    </div>

                    {/* Key Metrics & Significance Pill */}
                    <div className="flex items-center gap-3 self-end sm:self-center">
                      <div className="text-right">
                        <div className="text-xs font-semibold text-white">
                          {tx.shares_sold?.toLocaleString()} shares
                        </div>
                        <div className="text-[11px] font-mono text-emerald-400">
                          {formatCurrency(tx.estimated_value)}
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-1">
                        <span
                          className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${
                            isSignificant
                              ? 'bg-red-500/20 text-red-300 border-red-500/40'
                              : isReview
                              ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                              : 'bg-slate-800 text-slate-400 border-slate-700'
                          }`}
                        >
                          {tx.significance_category === 'significant'
                            ? 'Significant'
                            : tx.significance_category === 'requires_review'
                            ? 'Requires Review'
                            : 'Routine'}
                        </span>
                        
                        {/* 10b5-1 Indicator Badge */}
                        <span
                          className={`text-[9px] font-medium px-1.5 py-0.5 rounded ${
                            tx.is_10b5_1
                              ? 'bg-slate-800 text-slate-300 border border-slate-700'
                              : 'bg-rose-950/60 text-rose-300 border border-rose-800/40'
                          }`}
                        >
                          {tx.is_10b5_1 ? 'Rule 10b5-1 Plan' : 'Discretionary'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quantitative Holdings & Risk Bar */}
                  <div className="mt-3 pt-2.5 border-t border-slate-800/60 grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                    <div>
                      <span className="text-slate-500 text-[10px] block">Holding Disposed</span>
                      <span className="font-mono font-semibold text-slate-200">
                        {tx.percentage_holdings_sold != null
                          ? `${tx.percentage_holdings_sold.toFixed(1)}%`
                          : 'Not reported'}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-500 text-[10px] block">Remaining Shares</span>
                      <span className="font-mono text-slate-300">
                        {tx.shares_owned_after != null
                          ? tx.shares_owned_after.toLocaleString()
                          : 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-500 text-[10px] block">Execution Price</span>
                      <span className="font-mono text-slate-300">
                        {tx.price_per_share ? `$${tx.price_per_share.toFixed(2)}` : 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span className="text-slate-500 text-[10px] block">Risk Score</span>
                      <div className="flex items-center gap-1.5">
                        <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                          <div
                            className={`h-full ${
                              tx.risk_score >= 0.7
                                ? 'bg-red-500'
                                : tx.risk_score >= 0.4
                                ? 'bg-amber-500'
                                : 'bg-blue-500'
                            }`}
                            style={{ width: `${Math.min(100, (tx.risk_score || 0) * 100)}%` }}
                          />
                        </div>
                        <span className="font-mono text-[10px] text-slate-400">
                          {tx.risk_score?.toFixed(2) || '0.00'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quick Action & Forensics Drawer Toggle */}
                  <div className="mt-3 flex items-center justify-between pt-2 border-t border-slate-800/60">
                    <button
                      onClick={() => toggleExpand(idx)}
                      className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
                    >
                      {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      <span>{isExpanded ? 'Hide Forensics & EDGAR Link' : 'View Risk Breakdown & EDGAR Link'}</span>
                    </button>

                    {onEvaluateTransaction && (
                      <button
                        onClick={() => handleForecastClick(tx)}
                        className="px-2.5 py-1 text-xs font-semibold bg-blue-600/90 hover:bg-blue-500 text-white rounded flex items-center gap-1.5 shadow-sm transition-all"
                      >
                        <Sparkles className="h-3 w-3" />
                        Forecast Short Impact
                        <ArrowRight className="h-3 w-3" />
                      </button>
                    )}
                  </div>

                  {/* Expanded Forensic Analysis Drawer */}
                  {isExpanded && (
                    <div className="mt-3 p-3 bg-slate-950/80 rounded border border-slate-800 text-xs space-y-2.5 animate-fadeIn">
                      <div>
                        <span className="text-slate-400 font-semibold block mb-1">Analytical Takeaway:</span>
                        <p className="text-slate-300 italic">{tx.investor_takeaway}</p>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 border-t border-slate-800/80">
                        {/* Risk Factors */}
                        <div>
                          <span className="text-red-400 font-semibold text-[11px] block mb-1">
                            Elevated Risk Factors:
                          </span>
                          {tx.risk_factors && tx.risk_factors.length > 0 ? (
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-[11px]">
                              {tx.risk_factors.map((rf, rIdx) => (
                                <li key={rIdx}>{rf}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-slate-500 text-[11px]">No high-risk factors identified.</span>
                          )}
                        </div>

                        {/* Mitigating Factors */}
                        <div>
                          <span className="text-emerald-400 font-semibold text-[11px] block mb-1">
                            Mitigating Factors:
                          </span>
                          {tx.mitigating_factors && tx.mitigating_factors.length > 0 ? (
                            <ul className="list-disc list-inside space-y-0.5 text-slate-400 text-[11px]">
                              {tx.mitigating_factors.map((mf, mIdx) => (
                                <li key={mIdx}>{mf}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-slate-500 text-[11px]">No mitigating factors detected.</span>
                          )}
                        </div>
                      </div>

                      {/* SEC EDGAR Filing Link */}
                      <div className="pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px]">
                        <span className="text-slate-500">Source: Official SEC EDGAR Database (Form 4)</span>
                        {tx.filing_url ? (
                          <a
                            href={tx.filing_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 hover:text-blue-300 flex items-center gap-1 font-mono"
                          >
                            <FileText className="h-3 w-3" />
                            View SEC EDGAR Filing
                            <ExternalLink className="h-2.5 w-2.5" />
                          </a>
                        ) : (
                          <span className="text-slate-600 font-mono">EDGAR URL Unavailable</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Institutional Disclaimer */}
      <div className="p-2.5 rounded-lg bg-slate-950/40 border border-slate-800/60 text-[11px] text-slate-500 leading-relaxed">
        <span className="font-semibold text-slate-400">Institutional Disclaimer:</span> Form 4 insider transactions are 
        retrieved from public SEC EDGAR records. Dispositions routinely occur for pre-scheduled portfolio diversification, 
        Rule 10b5-1 tax withholding, and option expiration. These evaluations serve as quantitative research signals 
        and do not constitute automated trading instructions or financial advice.
      </div>
    </div>
  );
}
