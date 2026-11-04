import React, { useState, useEffect } from 'react';
import { 
  Bell, 
  ArrowRight, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp, 
  Search, 
  FileText,
  RefreshCw,
  Sparkles
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
    <div className="bg-[#242720] border border-[#393C33] rounded p-6 shadow-sm space-y-5">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-[#393C33] gap-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono text-[#A18B62] uppercase tracking-wider">
              REGULATORY SURVEILLANCE
            </span>
            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded border border-[#393C33] bg-[#1D201B] text-[#829477]">
              EDGAR XML ACTIVE
            </span>
          </div>
          <h3 className="text-base font-semibold text-[#E8E4D9] mt-0.5 font-mono">
            Executive Insider Selling & SEC Form 4 Intelligence
          </h3>
          <p className="text-xs text-[#85877C] mt-0.5">
            Automated screening of officer share dispositions, Rule 10b5-1 pre-scheduled plans, and liquidation forensics.
          </p>
        </div>

        {/* Tab Toggle */}
        <div className="flex items-center gap-1.5 p-1 bg-[#1D201B] border border-[#393C33] rounded self-start sm:self-center">
          <button
            onClick={() => setActiveTab('screener')}
            className={`px-3 py-1 text-xs font-mono rounded transition-colors ${
              activeTab === 'screener'
                ? 'bg-[#2B2E27] text-[#E8E4D9] border border-[#393C33]'
                : 'text-[#85877C] hover:text-[#AAA99D]'
            }`}
          >
            TICKER SCREENER
          </button>
          <button
            onClick={() => setActiveTab('alerts')}
            className={`px-3 py-1 text-xs font-mono rounded flex items-center gap-1.5 transition-colors ${
              activeTab === 'alerts'
                ? 'bg-[#2B2E27] text-[#A18B62] border border-[#393C33]'
                : 'text-[#85877C] hover:text-[#AAA99D]'
            }`}
          >
            <Bell className="h-3 w-3" />
            ALERT FEED ({alerts.length})
          </button>
        </div>
      </div>

      {/* Subheader info & Ticker Navigation */}
      {activeTab === 'screener' && (
        <div className="flex flex-wrap items-center justify-between gap-3 bg-[#1D201B] p-3 rounded border border-[#393C33]">
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs text-[#85877C] font-mono mr-1">COVERAGE:</span>
            {COVERAGE_TICKERS.map((sym) => (
              <button
                key={sym}
                onClick={() => handleTickerSelect(sym)}
                className={`px-2.5 py-1 text-xs font-mono rounded transition-colors ${
                  selectedTicker === sym
                    ? 'bg-[#A18B62] text-[#171916] font-bold'
                    : 'bg-[#242720] text-[#AAA99D] hover:text-[#E8E4D9] border border-[#393C33]'
                }`}
              >
                ${sym}
              </button>
            ))}
          </div>

          <form onSubmit={handleCustomSubmit} className="flex items-center gap-1.5">
            <input
              type="text"
              placeholder="LOOKUP TICKER..."
              value={customTicker}
              onChange={(e) => setCustomTicker(e.target.value)}
              className="px-2.5 py-1 text-xs bg-[#242720] border border-[#393C33] rounded text-[#E8E4D9] uppercase font-mono w-32 focus:outline-none focus:border-[#A18B62]"
            />
            <button
              type="submit"
              className="px-2.5 py-1 text-xs bg-[#2B2E27] text-[#AAA99D] hover:text-[#E8E4D9] border border-[#393C33] rounded flex items-center gap-1 font-mono"
            >
              <Search className="h-3 w-3" />
              SCAN
            </button>
          </form>
        </div>
      )}

      {/* Loading & Error States */}
      {isLoading && (
        <div className="py-8 flex flex-col items-center justify-center text-[#85877C] text-xs font-mono">
          <RefreshCw className="h-4 w-4 animate-spin text-[#A18B62] mb-2" />
          PARSING SEC EDGAR XML OWNERSHIP SUBMISSIONS...
        </div>
      )}

      {error && (
        <div className="p-3 bg-[#B66F58]/10 border border-[#B66F58]/40 rounded text-xs text-[#B66F58] flex items-center gap-2 font-mono">
          <span>{error}</span>
        </div>
      )}

      {/* Filings / Alerts List */}
      {!isLoading && !error && displayedList.length === 0 && (
        <div className="py-6 text-center text-[#85877C] text-xs font-mono bg-[#1D201B] rounded border border-[#393C33]">
          NO SIGNIFICANT INSIDER DISPOSITIONS FOUND MATCHING CRITERIA.
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
                className={`rounded border transition-colors ${
                  isSignificant
                    ? 'bg-[#1D201B] border-[#B66F58]/50'
                    : isReview
                    ? 'bg-[#1D201B] border-[#C49A58]/40'
                    : 'bg-[#1D201B] border-[#393C33]'
                }`}
              >
                <div className="p-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    {/* Insider info */}
                    <div className="flex items-start gap-3">
                      <div className="mt-1">
                        <span className={`h-2.5 w-2.5 rounded-full inline-block ${
                          isSignificant ? 'bg-[#B66F58]' : isReview ? 'bg-[#C49A58]' : 'bg-[#829477]'
                        }`} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="font-semibold text-[#E8E4D9] text-sm">{tx.insider_name}</span>
                          <span className="text-xs text-[#85877C]">({tx.insider_role})</span>
                          {tx.is_c_suite && (
                            <span className="text-[10px] font-mono px-1.5 py-0.2 rounded border border-[#A18B62]/40 bg-[#A18B62]/10 text-[#A18B62]">
                              C-SUITE
                            </span>
                          )}
                          <span className="font-mono text-xs font-bold text-[#A18B62]">${tx.ticker}</span>
                        </div>
                        <div className="flex items-center gap-3 text-[11px] text-[#85877C] font-mono mt-1">
                          <span>FILING: {tx.filing_date || 'N/A'}</span>
                          <span>&bull;</span>
                          <span>TX: {tx.transaction_date || 'N/A'}</span>
                          {tx.data_source && (
                            <>
                              <span>&bull;</span>
                              <span className="text-[#AAA99D] uppercase">SOURCE: {tx.data_source.replace(/_/g, ' ')}</span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Key Metrics & Significance Pill */}
                    <div className="flex items-center gap-3 self-end sm:self-center">
                      <div className="text-right font-mono">
                        <div className="text-xs font-semibold text-[#E8E4D9]">
                          {tx.shares_sold?.toLocaleString()} SHARES
                        </div>
                        <div className="text-[11px] text-[#A18B62]">
                          {formatCurrency(tx.estimated_value)}
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-1 font-mono">
                        <span
                          className={`text-[10px] font-bold uppercase px-2 py-0.5 rounded border ${
                            isSignificant
                              ? 'bg-[#B66F58]/20 text-[#B66F58] border-[#B66F58]/50'
                              : isReview
                              ? 'bg-[#C49A58]/20 text-[#C49A58] border-[#C49A58]/50'
                              : 'bg-[#2B2E27] text-[#85877C] border-[#393C33]'
                          }`}
                        >
                          {tx.significance_category === 'significant'
                            ? 'SIGNIFICANT'
                            : tx.significance_category === 'requires_review'
                            ? 'REQUIRES REVIEW'
                            : 'ROUTINE'}
                        </span>
                        
                        {/* 10b5-1 Indicator Badge */}
                        <span
                          className={`text-[9px] px-1.5 py-0.2 rounded border ${
                            tx.is_10b5_1
                              ? 'bg-[#242720] text-[#829477] border-[#829477]/40'
                              : 'bg-[#242720] text-[#B66F58] border-[#B66F58]/40'
                          }`}
                        >
                          {tx.is_10b5_1 ? 'RULE 10B5-1' : 'DISCRETIONARY'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quantitative Holdings & Risk Bar */}
                  <div className="mt-3 pt-3 border-t border-[#393C33] grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                    <div>
                      <span className="text-[#85877C] text-[10px] block">HOLDING DISPOSED</span>
                      <span className="font-semibold text-[#E8E4D9]">
                        {tx.percentage_holdings_sold != null
                          ? `${tx.percentage_holdings_sold.toFixed(1)}%`
                          : 'Not reported'}
                      </span>
                    </div>

                    <div>
                      <span className="text-[#85877C] text-[10px] block">REMAINING SHARES</span>
                      <span className="text-[#AAA99D]">
                        {tx.shares_owned_after != null
                          ? tx.shares_owned_after.toLocaleString()
                          : 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span className="text-[#85877C] text-[10px] block">EXECUTION PRICE</span>
                      <span className="text-[#AAA99D]">
                        {tx.price_per_share ? `$${tx.price_per_share.toFixed(2)}` : 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span className="text-[#85877C] text-[10px] block">RISK SCORE</span>
                      <div className="flex items-center gap-1.5">
                        <div className="flex-1 bg-[#242720] border border-[#393C33] rounded h-1.5 overflow-hidden">
                          <div
                            className={`h-full ${
                              tx.risk_score >= 0.7
                                ? 'bg-[#B66F58]'
                                : tx.risk_score >= 0.4
                                ? 'bg-[#C49A58]'
                                : 'bg-[#A18B62]'
                            }`}
                            style={{ width: `${Math.min(100, (tx.risk_score || 0) * 100)}%` }}
                          />
                        </div>
                        <span className="text-[10px] text-[#AAA99D]">
                          {tx.risk_score?.toFixed(2) || '0.00'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Quick Action & Forensics Drawer Toggle */}
                  <div className="mt-3 flex items-center justify-between pt-2 border-t border-[#393C33]">
                    <button
                      onClick={() => toggleExpand(idx)}
                      className="text-xs font-mono text-[#85877C] hover:text-[#E8E4D9] flex items-center gap-1 transition-colors"
                    >
                      {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                      <span>{isExpanded ? 'HIDE FORENSICS & EDGAR LINK' : 'VIEW RISK BREAKDOWN & EDGAR LINK'}</span>
                    </button>

                    {onEvaluateTransaction && (
                      <button
                        onClick={() => handleForecastClick(tx)}
                        className="px-2.5 py-1 text-xs font-mono font-medium rounded border border-[#A18B62] bg-[#A18B62]/10 hover:bg-[#A18B62] text-[#A18B62] hover:text-[#171916] flex items-center gap-1.5 transition-colors"
                      >
                        <Sparkles className="h-3 w-3" />
                        FORECAST SHORT IMPACT
                        <ArrowRight className="h-3 w-3" />
                      </button>
                    )}
                  </div>

                  {/* Expanded Forensic Analysis Drawer */}
                  {isExpanded && (
                    <div className="mt-3 p-3 bg-[#242720] rounded border border-[#393C33] text-xs space-y-2.5 animate-fade-in">
                      <div>
                        <span className="text-[#85877C] font-mono text-[10px] uppercase block mb-1">
                          ANALYTICAL TAKEAWAY
                        </span>
                        <p className="text-[#E8E4D9] italic leading-relaxed">{tx.investor_takeaway}</p>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2 border-t border-[#393C33]">
                        {/* Risk Factors */}
                        <div>
                          <span className="text-[#B66F58] font-mono text-[10px] uppercase block mb-1">
                            ELEVATED RISK FACTORS
                          </span>
                          {tx.risk_factors && tx.risk_factors.length > 0 ? (
                            <ul className="list-disc list-inside space-y-0.5 text-[#AAA99D] text-[11px]">
                              {tx.risk_factors.map((rf, rIdx) => (
                                <li key={rIdx}>{rf}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-[#85877C] text-[11px] font-mono">No elevated risk factors detected.</span>
                          )}
                        </div>

                        {/* Mitigating Factors */}
                        <div>
                          <span className="text-[#829477] font-mono text-[10px] uppercase block mb-1">
                            MITIGATING FACTORS
                          </span>
                          {tx.mitigating_factors && tx.mitigating_factors.length > 0 ? (
                            <ul className="list-disc list-inside space-y-0.5 text-[#AAA99D] text-[11px]">
                              {tx.mitigating_factors.map((mf, mIdx) => (
                                <li key={mIdx}>{mf}</li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-[#85877C] text-[11px] font-mono">No mitigating factors detected.</span>
                          )}
                        </div>
                      </div>

                      {/* SEC EDGAR Filing Link */}
                      <div className="pt-2 border-t border-[#393C33] flex items-center justify-between text-[11px] font-mono">
                        <span className="text-[#85877C]">SOURCE: OFFICIAL SEC EDGAR REPOSITORY</span>
                        {tx.filing_url ? (
                          <a
                            href={tx.filing_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[#A18B62] hover:text-[#B29B70] flex items-center gap-1"
                          >
                            <FileText className="h-3 w-3" />
                            VIEW SEC FORM 4 FILING
                            <ExternalLink className="h-2.5 w-2.5" />
                          </a>
                        ) : (
                          <span className="text-[#85877C]">EDGAR URL UNAVAILABLE</span>
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
      <div className="p-3 rounded bg-[#1D201B] border border-[#393C33] text-[11px] text-[#85877C] leading-relaxed font-mono">
        <strong className="text-[#AAA99D]">INSTITUTIONAL DISCLAIMER:</strong> Form 4 insider transactions are retrieved 
        from public SEC EDGAR records. Dispositions occur routinely for pre-scheduled portfolio diversification, 
        Rule 10b5-1 tax withholding, and option expiration. These evaluations serve as quantitative research signals 
        and do not constitute automated trading instructions or investment advice.
      </div>
    </div>
  );
}
