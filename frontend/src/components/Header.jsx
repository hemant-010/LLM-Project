import React, { useState, useEffect, useRef } from 'react';
import { 
  Landmark, 
  Bell, 
  Check, 
  Trash2, 
  X,
  FileText,
  AlertTriangle,
  Info
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || '';

export default function Header({ backendHealth, isCheckingHealth, onSelectNotification }) {
  const isHealthy = backendHealth?.status === 'healthy';
  const models = backendHealth?.models || {};

  const [notifications, setNotifications] = useState([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Fetch persistent notifications
  const fetchNotifications = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications?limit=25`);
      if (res.ok) {
        const data = await res.json();
        setNotifications(data.notifications || []);
        setUnreadCount(data.unread_count || 0);
      }
    } catch (err) {
      console.error('Failed to fetch notifications:', err);
    }
  };

  useEffect(() => {
    fetchNotifications();
    const interval = setInterval(fetchNotifications, 15000);
    return () => clearInterval(interval);
  }, []);

  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Mark single or all as read
  const handleMarkAsRead = async (notifId = null) => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications/mark-read`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notification_id: notifId })
      });
      if (res.ok) {
        const data = await res.json();
        setUnreadCount(data.unread_count || 0);
        setNotifications((prev) =>
          prev.map((n) => (!notifId || n.id === notifId ? { ...n, is_read: true } : n))
        );
      }
    } catch (err) {
      console.error('Failed to mark read:', err);
    }
  };

  // Clear all notifications
  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/notifications`, {
        method: 'DELETE'
      });
      if (res.ok) {
        setNotifications([]);
        setUnreadCount(0);
      }
    } catch (err) {
      console.error('Failed to clear notifications:', err);
    }
  };

  const handleNotificationClick = (notif) => {
    if (!notif.is_read) {
      handleMarkAsRead(notif.id);
    }
    if (onSelectNotification) {
      onSelectNotification(notif);
      setIsOpen(false);
    }
  };

  return (
    <header className="border-b border-[#393C33] bg-[#171916]/95 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand Identity */}
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded border border-[#393C33] bg-[#242720] flex items-center justify-center text-[#A18B62]">
            <Landmark className="h-4 w-4" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm font-semibold tracking-wider text-[#E8E4D9] uppercase font-mono">
                Short Interest Intelligence
              </h1>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-[#393C33] bg-[#242720] text-[#A18B62]">
                PHASE 2
              </span>
            </div>
            <p className="text-[11px] text-[#85877C]">
              Institutional Market Catalyst & FINRA Settlement Prediction Terminal
            </p>
          </div>
        </div>

        {/* Status Ledgers & Notification Center */}
        <div className="flex items-center flex-wrap gap-2 text-xs">
          {/* Backend Connection */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-[#393C33] bg-[#1D201B]">
            <span className={`h-1.5 w-1.5 rounded-full ${isHealthy ? 'bg-[#829477]' : 'bg-[#B66F58]'}`} />
            <span className="text-[11px] text-[#AAA99D] font-mono">
              API: {isHealthy ? 'ONLINE' : 'CONNECTING'}
            </span>
          </div>

          {/* LR Model Status */}
          <div className="px-2.5 py-1 rounded border border-[#393C33] bg-[#1D201B] text-[11px] font-mono">
            <span className="text-[#85877C] mr-1">LR:</span>
            <span className="text-[#E8E4D9]">
              {models.logistic_regression === 'ready' ? 'READY (30k TF-IDF)' : 'OFFLINE'}
            </span>
          </div>

          {/* XGBoost Status */}
          <div className="px-2.5 py-1 rounded border border-[#393C33] bg-[#1D201B] text-[11px] font-mono">
            <span className="text-[#85877C] mr-1">XGB:</span>
            <span className="text-[#E8E4D9]">
              {models.xgboost === 'ready' ? 'READY (TREES)' : 'OFFLINE'}
            </span>
          </div>

          {/* Gemini Mode */}
          <div className="px-2.5 py-1 rounded border border-[#393C33] bg-[#1D201B] text-[11px] font-mono">
            <span className="text-[#85877C] mr-1">ENRICHMENT:</span>
            <span className={backendHealth?.environment?.gemini_configured ? 'text-[#A18B62]' : 'text-[#85877C]'}>
              {backendHealth?.environment?.gemini_configured ? 'GEMINI' : 'HEURISTIC'}
            </span>
          </div>

          {/* Persistent Investor Notification Bell */}
          <div className="relative ml-1" ref={dropdownRef}>
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="relative p-1.5 rounded border border-[#393C33] bg-[#242720] hover:bg-[#2B2E27] text-[#AAA99D] hover:text-[#E8E4D9] transition-colors"
              title="Investor Alerts & Notifications"
            >
              <Bell className="h-4 w-4" />
              {unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 h-3.5 min-w-[14px] px-1 bg-[#B66F58] text-[#E8E4D9] font-bold text-[9px] rounded-full flex items-center justify-center">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </button>

            {/* Notification Drawer */}
            {isOpen && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 bg-[#242720] border border-[#393C33] rounded shadow-2xl z-50 overflow-hidden animate-fade-in">
                {/* Header */}
                <div className="p-3 bg-[#1D201B] border-b border-[#393C33] flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-semibold text-[#E8E4D9] uppercase tracking-wide">
                      Investor Notifications
                    </span>
                    {unreadCount > 0 && (
                      <span className="text-[10px] font-mono bg-[#B66F58]/20 text-[#B66F58] px-1.5 py-0.2 rounded border border-[#B66F58]/40">
                        {unreadCount} UNREAD
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {unreadCount > 0 && (
                      <button
                        onClick={() => handleMarkAsRead(null)}
                        className="text-[10px] text-[#A18B62] hover:text-[#B29B70] flex items-center gap-1 font-mono"
                      >
                        <Check className="h-3 w-3" />
                        MARK ALL READ
                      </button>
                    )}
                    <button
                      onClick={() => setIsOpen(false)}
                      className="text-[#85877C] hover:text-[#E8E4D9] p-0.5"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                {/* Notifications List */}
                <div className="max-h-80 overflow-y-auto divide-y divide-[#393C33]">
                  {notifications.length === 0 ? (
                    <div className="py-8 text-center text-xs text-[#85877C]">
                      No active investor alerts.
                    </div>
                  ) : (
                    notifications.map((notif) => {
                      const isSignificant = notif.severity === 'significant';
                      const isReview = notif.severity === 'requires_review';

                      return (
                        <div
                          key={notif.id}
                          onClick={() => handleNotificationClick(notif)}
                          className={`p-3 transition-colors cursor-pointer flex items-start gap-2.5 ${
                            !notif.is_read
                              ? 'bg-[#2B2E27]/80 hover:bg-[#2B2E27]'
                              : 'hover:bg-[#1D201B]'
                          }`}
                        >
                          <div className="mt-0.5 shrink-0">
                            {isSignificant ? (
                              <span className="h-2 w-2 rounded-full bg-[#B66F58] inline-block" />
                            ) : isReview ? (
                              <span className="h-2 w-2 rounded-full bg-[#C49A58] inline-block" />
                            ) : (
                              <span className="h-2 w-2 rounded-full bg-[#829477] inline-block" />
                            )}
                          </div>

                          <div className="flex-1 min-w-0">
                            <div className="flex items-center justify-between gap-1 mb-1">
                              <span className="font-medium text-xs text-[#E8E4D9] truncate">
                                {notif.title}
                              </span>
                              {!notif.is_read && (
                                <span className="h-1.5 w-1.5 rounded-full bg-[#A18B62] shrink-0" />
                              )}
                            </div>
                            <p className="text-[11px] text-[#AAA99D] line-clamp-2 leading-relaxed">
                              {notif.message}
                            </p>
                            <div className="flex items-center gap-2 mt-1.5 text-[10px] text-[#85877C] font-mono">
                              {notif.ticker && (
                                <span className="text-[#A18B62]">
                                  ${notif.ticker}
                                </span>
                              )}
                              <span>&bull;</span>
                              <span>
                                {new Date(notif.created_at).toLocaleDateString(undefined, {
                                  month: 'short',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>

                {/* Footer */}
                {notifications.length > 0 && (
                  <div className="p-2.5 bg-[#1D201B] border-t border-[#393C33] flex items-center justify-between text-[10px] text-[#85877C] font-mono">
                    <span>PERSISTENT SQLITE REPOSITORY</span>
                    <button
                      onClick={handleClearAll}
                      className="text-[#85877C] hover:text-[#B66F58] flex items-center gap-1 transition-colors"
                    >
                      <Trash2 className="h-3 w-3" />
                      CLEAR ALL
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
