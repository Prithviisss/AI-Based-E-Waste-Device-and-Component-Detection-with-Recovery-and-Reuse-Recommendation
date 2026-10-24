import React, { useState, useEffect, useRef } from 'react';
import {
  Upload, Sparkles, Cpu, Recycle, DollarSign, Leaf, AlertTriangle,
  CheckCircle, ExternalLink, Activity, Info, X, ChevronRight, BarChart2,
  FileText, Eye, ShieldCheck, RefreshCw, Zap
} from 'lucide-react';
import './App.css';

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [age, setAge] = useState(3);
  const [working, setWorking] = useState('');
  const [budget, setBudget] = useState(20);
  const [isDragOver, setIsDragOver] = useState(false);

  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(1);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const [activeTab, setActiveTab] = useState('components');
  const [healthStatus, setHealthStatus] = useState('checking');
  const [showMetricsModal, setShowMetricsModal] = useState(false);
  const [metricsData, setMetricsData] = useState(null);

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        setHealthStatus('online');
      } else {
        setHealthStatus('offline');
      }
    } catch {
      setHealthStatus('offline');
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/metrics');
      if (res.ok) {
        const data = await res.json();
        setMetricsData(data);
      }
    } catch (e) {
      console.error("Failed to fetch metrics", e);
    }
    setShowMetricsModal(true);
  };

  const handleFileChange = (newFile) => {
    if (!newFile) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(newFile.type)) {
      setError('Please upload a valid JPEG, PNG, or WebP image.');
      return;
    }
    setError(null);
    setFile(newFile);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(newFile));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const clearSelectedFile = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Load sample demo image
  const loadDemoImage = async (type) => {
    setError(null);
    try {
      const path = type === 'pcb' ? '/demo/sample_pcb.jpg' : '/demo/sample_ramssd.jpg';
      const name = type === 'pcb' ? 'sample_pcb.jpg' : 'sample_ramssd.jpg';
      const res = await fetch(path);
      if (!res.ok) throw new Error('Demo asset not found on server.');
      const blob = await res.blob();
      const demoFile = new File([blob], name, { type: 'image/jpeg' });
      handleFileChange(demoFile);
    } catch (err) {
      setError('Failed to load demo image: ' + err.message);
    }
  };

  const runAnalysis = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setLoadingStage(1);

    const stageTimer1 = setTimeout(() => setLoadingStage(2), 700);
    const stageTimer2 = setTimeout(() => setLoadingStage(3), 1400);
    const stageTimer3 = setTimeout(() => setLoadingStage(4), 2100);

    const formData = new FormData();
    formData.append('image', file);
    if (age !== '') formData.append('age_years', age);
    if (working !== '') formData.append('working', working);
    formData.append('budget_usd', budget);

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorJson.detail || 'Analysis request failed.');
      }

      const data = await response.json();
      setResults(data);
      setTimeout(() => {
        if (resultsRef.current) {
          resultsRef.current.scrollIntoView({ behavior: 'smooth' });
        }
      }, 100);
    } catch (err) {
      setError(err.message || 'Connection to analysis backend failed.');
    } finally {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      clearTimeout(stageTimer3);
      setLoading(false);
    }
  };

  const getRouteBadge = (route) => {
    if (!route) return <span className="badge">Unknown</span>;
    if (route === 'direct_reuse') return <span className="badge badge-green"><Recycle size={12} /> Direct Reuse</span>;
    if (route === 'refurbishment') return <span className="badge badge-yellow"><Zap size={12} /> Refurbish</span>;
    return <span className="badge badge-blue"><Cpu size={12} /> Materials</span>;
  };

  return (
    <div className="app-wrapper">
      {/* ── Navigation ──────────────────────────────────────────────── */}
      <nav className="navbar">
        <div className="container nav-container">
          <div className="brand" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
            <div className="brand-icon-wrap">
              <Recycle size={22} />
            </div>
            <div className="brand-title">
              E-Waste <strong>RecoveryAI</strong>
            </div>
          </div>

          <div className="nav-actions">
            <div className="health-pill" title="FastAPI Backend Status">
              <span className={`status-dot ${healthStatus}`} />
              <span>{healthStatus === 'online' ? 'API Active' : 'API Offline'}</span>
            </div>

            <button 
              className="btn btn-secondary" 
              style={{ padding: '0.45rem 0.9rem', fontSize: '0.85rem' }}
              onClick={fetchMetrics}
            >
              <BarChart2 size={15} />
              <span>Model Metrics</span>
            </button>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ────────────────────────────────────────────── */}
      <header className="hero-section">
        <div className="container">
          <div className="hero-pill">
            <Sparkles size={14} />
            <span>Dual YOLOv8 + EfficientNet-B0 + MILP Optimization</span>
          </div>

          <h1 className="hero-title">
            Intelligent E-Waste Identification &amp; <br />
            <span className="gradient-text">Recovery Pathway Optimizer</span>
          </h1>

          <p className="hero-description">
            Upload an electronic device or motherboard photo. Our computer vision pipeline
            identifies the device, localizes internal components, and calculates optimal
            circular economy reuse pathways with real-world CO₂ and financial valuation.
          </p>

          <div className="stats-ticker">
            <div className="ticker-item">
              <span className="ticker-val gradient-eco">12 Categories</span>
              <span className="ticker-lbl">Device Classifier</span>
            </div>
            <div className="ticker-item">
              <span className="ticker-val gradient-text">2x Detectors</span>
              <span className="ticker-lbl">PCB &amp; RAM/SSD YOLOv8</span>
            </div>
            <div className="ticker-item">
              <span className="ticker-val" style={{ color: '#38bdf8' }}>MILP Solver</span>
              <span className="ticker-lbl">Optimal Budget Plan</span>
            </div>
            <div className="ticker-item">
              <span className="ticker-val" style={{ color: '#fbbf24' }}>UN / WEEE</span>
              <span className="ticker-lbl">Cited Knowledge Base</span>
            </div>
          </div>
        </div>
      </header>

      {/* ── Workspace Section ───────────────────────────────────────── */}
      <main className="container workspace-section">
        <div className="glass-panel workspace-card">
          <div className="section-heading-group">
            <h2><Cpu size={22} className="gradient-text" /> Device Visual Analysis</h2>
            <p>Upload a clear photo or select a verified test sample to run the multi-stage AI pipeline.</p>
          </div>

          {/* Upload Dropzone / Preview */}
          {!previewUrl ? (
            <div
              className={`dropzone-container ${isDragOver ? 'is-dragover' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current && fileInputRef.current.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                style={{ display: 'none' }}
                onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
              />
              <div className="dz-icon-halo">
                <Upload size={32} />
              </div>
              <div className="dz-main-text">Drag &amp; drop e-waste photo here</div>
              <div className="dz-sub-text">Supports JPEG, PNG, WebP (high-resolution recommended)</div>
              <button 
                type="button" 
                className="btn btn-secondary" 
                style={{ marginTop: '1rem', pointerEvents: 'none' }}
              >
                Browse Files
              </button>
            </div>
          ) : (
            <div className="preview-card">
              <img src={previewUrl} alt="E-waste specimen preview" />
              <div className="preview-overlay">
                <button className="btn btn-secondary" onClick={clearSelectedFile} title="Remove image">
                  <X size={16} /> Remove
                </button>
              </div>
              <div className="preview-info">
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.95rem' }}>{file?.name}</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    {file ? `${(file.size / 1024).toFixed(1)} KB` : ''}
                  </div>
                </div>
                <span className="badge badge-green"><CheckCircle size={12} /> Specimen Ready</span>
              </div>
            </div>
          )}

          {/* Quick Demo Selector */}
          <div className="demo-bar">
            <span className="demo-label">Quick Test Presets:</span>
            <button className="demo-chip" type="button" onClick={() => loadDemoImage('pcb')}>
              <Cpu size={14} /> Sample PCB Board
            </button>
            <button className="demo-chip" type="button" onClick={() => loadDemoImage('ramssd')}>
              <Zap size={14} /> Sample RAM/SSD Specimen
            </button>
          </div>

          {/* Interactive Parameters Panel */}
          <div className="parameters-grid">
            {/* Age */}
            <div className="param-group">
              <label className="param-label">
                <span>Estimated Device Age</span>
                <span className="param-value-tag">{age} Years</span>
              </label>
              <input
                type="range"
                min="0"
                max="25"
                value={age}
                onChange={(e) => setAge(parseInt(e.target.value) || 0)}
                style={{ accentColor: 'var(--primary)', cursor: 'pointer' }}
              />
              <input
                type="number"
                min="0"
                max="50"
                value={age}
                onChange={(e) => setAge(parseInt(e.target.value) || 0)}
                className="param-input"
                placeholder="Age in years"
              />
            </div>

            {/* Working Condition */}
            <div className="param-group">
              <label className="param-label">Operational Condition</label>
              <div className="segmented-control">
                <button
                  type="button"
                  className={`segmented-btn ${working === 'true' ? 'active working' : ''}`}
                  onClick={() => setWorking('true')}
                >
                  🟢 Working
                </button>
                <button
                  type="button"
                  className={`segmented-btn ${working === 'false' ? 'active broken' : ''}`}
                  onClick={() => setWorking('false')}
                >
                  🔴 Faulty
                </button>
                <button
                  type="button"
                  className={`segmented-btn ${working === '' ? 'active' : ''}`}
                  onClick={() => setWorking('')}
                >
                  ⚪ Unknown
                </button>
              </div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                Influences rule-based reuse vs salvage routing.
              </div>
            </div>

            {/* Budget */}
            <div className="param-group">
              <label className="param-label">
                <span>Recovery Budget (USD)</span>
                <span className="param-value-tag">${budget}</span>
              </label>
              <input
                type="number"
                min="0"
                step="5"
                value={budget}
                onChange={(e) => setBudget(parseFloat(e.target.value) || 0)}
                className="param-input"
                placeholder="20.00"
              />
              <div className="budget-presets">
                {[10, 20, 50, 100].map((val) => (
                  <button
                    key={val}
                    type="button"
                    className="budget-chip"
                    onClick={() => setBudget(val)}
                  >
                    ${val}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Action Button & Pipeline Progress */}
          <div className="action-box">
            <button
              type="button"
              className="btn btn-primary full-btn"
              disabled={!file || loading}
              onClick={runAnalysis}
            >
              {loading ? (
                <>
                  <RefreshCw size={20} className="animate-spin" />
                  <span>Processing Deep Neural Pipeline...</span>
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  <span>Execute AI Analysis &amp; Optimize Pathways</span>
                </>
              )}
            </button>

            {loading && (
              <div className="pipeline-progress-card animate-fade-in">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.88rem' }}>
                  <span style={{ fontWeight: 600, color: 'var(--primary-light)' }}>
                    Stage {loadingStage} of 4:
                  </span>
                  <span style={{ color: 'var(--text-secondary)' }}>
                    {loadingStage === 1 && 'EfficientNet-B0 Device Classification...'}
                    {loadingStage === 2 && 'Parallel YOLOv8 PCB + RAM/SSD Inference...'}
                    {loadingStage === 3 && 'Knowledge Base Rule Evaluation & Value Mapping...'}
                    {loadingStage === 4 && 'MILP Mixed-Integer Linear Optimization...'}
                  </span>
                </div>
                <div className="pipeline-steps">
                  <div className={`step-indicator ${loadingStage >= 1 ? 'active' : ''}`}>
                    <span>01</span>
                    <span>Classify Device</span>
                  </div>
                  <div className={`step-indicator ${loadingStage >= 2 ? 'active' : ''}`}>
                    <span>02</span>
                    <span>YOLOv8 Detect</span>
                  </div>
                  <div className={`step-indicator ${loadingStage >= 3 ? 'active' : ''}`}>
                    <span>03</span>
                    <span>KB Rules</span>
                  </div>
                  <div className={`step-indicator ${loadingStage >= 4 ? 'active' : ''}`}>
                    <span>04</span>
                    <span>MILP Solver</span>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="alert-card danger" style={{ marginTop: '1.25rem' }}>
                <AlertTriangle size={20} />
                <div>
                  <strong>Analysis Error:</strong> {error}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* ── Results Section ─────────────────────────────────────────── */}
        {results && (
          <div className="results-section animate-fade-in" ref={resultsRef}>
            <div className="results-header-banner">
              <div>
                <h2 style={{ fontSize: '1.85rem' }}>Analysis &amp; Recovery Intelligence</h2>
                <p style={{ color: 'var(--text-secondary)' }}>
                  Multi-model evaluation report for uploaded specimen.
                </p>
              </div>

              {results.report_url && (
                <a
                  href={results.report_url}
                  target="_blank"
                  rel="noreferrer"
                  className="btn btn-secondary"
                >
                  <FileText size={16} /> Open Full HTML Report <ExternalLink size={14} />
                </a>
              )}
            </div>

            {/* Metrics Summary Cards */}
            <div className="metrics-summary-grid">
              {/* Identified Device */}
              <div className="glass-panel stat-card">
                <div className="stat-card-top">
                  <span className="stat-title">Device Classification</span>
                  <Cpu size={18} color="var(--primary)" />
                </div>
                <div className="stat-value" style={{ textTransform: 'capitalize' }}>
                  {results.device || 'Unknown'}
                </div>
                <div>
                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{ width: `${(results.device_confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                  <div className="stat-subtitle">
                    {results.device_confidence ? (results.device_confidence * 100).toFixed(1) : 0}% classification confidence
                  </div>
                </div>
              </div>

              {/* Recovered Value */}
              <div className="glass-panel stat-card">
                <div className="stat-card-top">
                  <span className="stat-title">Optimized Value Yield</span>
                  <DollarSign size={18} color="#34d399" />
                </div>
                <div className="stat-value" style={{ color: '#34d399' }}>
                  ${results.optimized_pathways?.total_value_usd ?? '0.00'}
                </div>
                <div className="stat-subtitle">
                  Budget Spent: ${results.optimized_pathways?.total_cost_usd ?? '0.00'} / ${budget}
                </div>
              </div>

              {/* CO2 Saved */}
              <div className="glass-panel stat-card">
                <div className="stat-card-top">
                  <span className="stat-title">Carbon Offset</span>
                  <Leaf size={18} color="#06b6d4" />
                </div>
                <div className="stat-value" style={{ color: '#38bdf8' }}>
                  {results.optimized_pathways?.total_co2_saved_kg ?? '0.00'} <span style={{ fontSize: '1.1rem' }}>kg</span>
                </div>
                <div className="stat-subtitle">
                  Estimated avoided greenhouse gas emissions
                </div>
              </div>

              {/* Optimization Status */}
              <div className="glass-panel stat-card">
                <div className="stat-card-top">
                  <span className="stat-title">Optimization Solver</span>
                  <ShieldCheck size={18} color={results.optimized_pathways?.status === 'optimal' ? '#10b981' : '#f59e0b'} />
                </div>
                <div
                  className="stat-value"
                  style={{
                    fontSize: '1.4rem',
                    color: results.optimized_pathways?.status === 'optimal' ? '#34d399' : '#fbbf24',
                    textTransform: 'uppercase',
                  }}
                >
                  {results.optimized_pathways?.status || 'FEASIBLE'}
                </div>
                <div className="stat-subtitle">
                  {results.optimized_pathways?.status === 'optimal'
                    ? 'MILP global optimum achieved'
                    : results.optimized_pathways?.message || 'Constrained evaluation'}
                </div>
              </div>
            </div>

            {/* Warnings and Unmapped Elements */}
            {results.unmapped && results.unmapped.length > 0 && (
              <div className="alert-card warning">
                <AlertTriangle size={20} />
                <div>
                  <strong>Unmapped Component Detections:</strong>{' '}
                  {results.unmapped.join(', ')}. These components do not have a dedicated entry in the knowledge base.
                </div>
              </div>
            )}

            {/* Tabs Navigation */}
            <div className="tabs-navigation">
              <button
                className={`tab-btn ${activeTab === 'components' ? 'active' : ''}`}
                onClick={() => setActiveTab('components')}
              >
                <Cpu size={16} /> Detected Components ({results.detections?.length || 0})
              </button>
              <button
                className={`tab-btn ${activeTab === 'milp' ? 'active' : ''}`}
                onClick={() => setActiveTab('milp')}
              >
                <ShieldCheck size={16} /> MILP Pathway Plan
              </button>
              <button
                className={`tab-btn ${activeTab === 'gradcam' ? 'active' : ''}`}
                onClick={() => setActiveTab('gradcam')}
              >
                <Eye size={16} /> Explainability (Grad-CAM)
              </button>
              <button
                className={`tab-btn ${activeTab === 'raw' ? 'active' : ''}`}
                onClick={() => setActiveTab('raw')}
              >
                <FileText size={16} /> Raw API Payload
              </button>
            </div>

            {/* Tab 1: Components Table */}
            {activeTab === 'components' && (
              <div className="glass-panel" style={{ padding: '1.5rem', borderRadius: '16px' }}>
                {results.detections && results.detections.length > 0 ? (
                  <div className="table-responsive">
                    <table className="styled-table">
                      <thead>
                        <tr>
                          <th>Detected Component</th>
                          <th>KB Identifier</th>
                          <th>Confidence</th>
                          <th>Recommended Route</th>
                          <th>Est. Value</th>
                          <th>CO₂ Saved</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.detections.map((det, idx) => (
                          <tr key={idx}>
                            <td style={{ fontWeight: 600 }}>{det.label}</td>
                            <td>
                              <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
                                {det.kb_key || <span style={{ color: 'var(--warn)' }}>unmapped</span>}
                              </code>
                            </td>
                            <td>
                              <span style={{ fontFamily: 'var(--font-mono)' }}>
                                {(det.confidence * 100).toFixed(1)}%
                              </span>
                            </td>
                            <td>{getRouteBadge(det.route)}</td>
                            <td style={{ color: '#34d399', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                              {det.estimated_value_usd != null ? `$${det.estimated_value_usd}` : '—'}
                            </td>
                            <td style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                              {det.co2_saved_kg != null ? `${det.co2_saved_kg} kg` : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                    No discrete components localized by YOLOv8 at the current confidence threshold.
                  </div>
                )}
              </div>
            )}

            {/* Tab 2: MILP Pathway */}
            {activeTab === 'milp' && (
              <div className="glass-panel" style={{ padding: '1.75rem', borderRadius: '16px' }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '0.75rem' }}>
                  Mixed-Integer Linear Programming Optimization Results
                </h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '1.5rem' }}>
                  The optimizer evaluated all feasible recovery actions subject to your budget constraint of <strong>${budget}.00 USD</strong>.
                </p>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem' }}>
                  <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
                    <h4 style={{ color: 'var(--primary-light)', fontSize: '1rem', marginBottom: '0.75rem' }}>
                      Selected Recovery Actions
                    </h4>
                    {results.optimized_pathways?.selected?.length ? (
                      <ul style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                        {results.optimized_pathways.selected.map((item, idx) => (
                          <li key={idx} style={{ color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                            {item}
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>None selected within budget.</p>
                    )}
                  </div>

                  <div style={{ background: 'rgba(15, 23, 42, 0.7)', padding: '1.25rem', borderRadius: '12px', border: '1px solid var(--border-subtle)' }}>
                    <h4 style={{ color: '#38bdf8', fontSize: '1rem', marginBottom: '0.75rem' }}>
                      Economic &amp; Environmental Impact
                    </h4>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.9rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Net Financial Yield:</span>
                        <strong style={{ color: '#34d399' }}>${results.optimized_pathways?.total_value_usd ?? '0.00'}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Total Cost Expended:</span>
                        <strong>${results.optimized_pathways?.total_cost_usd ?? '0.00'}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-muted)' }}>Net CO₂ Avoided:</span>
                        <strong style={{ color: '#38bdf8' }}>{results.optimized_pathways?.total_co2_saved_kg ?? '0.00'} kg</strong>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tab 3: Grad-CAM */}
            {activeTab === 'gradcam' && (
              <div className="glass-panel" style={{ padding: '1.75rem', borderRadius: '16px' }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Grad-CAM Activation Visualizations</h3>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', marginBottom: '1.5rem' }}>
                  Gradient-weighted Class Activation Mapping displays visual attention regions of the neural network.
                </p>

                {results.heatmap_urls && results.heatmap_urls.length > 0 ? (
                  <div className="heatmaps-grid">
                    {results.heatmap_urls.map((h, idx) => (
                      <div key={idx} className="heatmap-card">
                        <img src={h.url} alt={h.label} />
                        <div className="heatmap-card-footer">
                          <span>{h.label}</span>
                          <span className="badge badge-blue">Attention Map</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)' }}>
                    Grad-CAM heatmap not generated for this inference pass.
                  </div>
                )}
              </div>
            )}

            {/* Tab 4: Raw JSON */}
            {activeTab === 'raw' && (
              <div className="glass-panel" style={{ padding: '1.5rem', borderRadius: '16px' }}>
                <pre
                  style={{
                    background: '#090d16',
                    padding: '1.25rem',
                    borderRadius: '8px',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                    color: '#a7f3d0',
                    overflowX: 'auto',
                    maxHeight: '420px',
                  }}
                >
                  {JSON.stringify(results, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ── System Architecture & How it works ──────────────────────── */}
      <section className="container" style={{ padding: '2rem 1.5rem 4rem' }}>
        <div className="disclaimer-banner">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            <Info size={16} /> Methodology Transparency Notice
          </div>
          <div style={{ marginTop: '0.35rem', fontSize: '0.82rem', color: '#fde68a' }}>
            Device classification &amp; component localization are driven by deep neural networks. 
            Recovery potential, material valuations, and carbon savings are calculated via our rule-based 
            knowledge base and MILP solver — no synthetic "accuracy" metric applies to economic recommendations.
          </div>
        </div>
      </section>

      {/* ── Metrics Modal ───────────────────────────────────────────── */}
      {showMetricsModal && (
        <div className="modal-overlay" onClick={() => setShowMetricsModal(false)}>
          <div className="glass-panel modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 style={{ fontSize: '1.3rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <BarChart2 size={20} className="gradient-text" /> Neural Model Evaluation Metrics
              </h3>
              <button className="btn-ghost" onClick={() => setShowMetricsModal(false)}>
                <X size={20} />
              </button>
            </div>

            {metricsData ? (
              <pre
                style={{
                  background: '#090d16',
                  padding: '1.25rem',
                  borderRadius: '10px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.85rem',
                  color: '#38bdf8',
                  overflowX: 'auto',
                  maxHeight: '450px',
                }}
              >
                {JSON.stringify(metricsData, null, 2)}
              </pre>
            ) : (
              <div style={{ textAlign: 'center', padding: '2rem' }}>Loading system metrics...</div>
            )}
          </div>
        </div>
      )}

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="footer">
        <div className="container">
          <p>
            E-Waste RecoveryAI &nbsp;•&nbsp; Powered by Kaggle, HuggingFace, &amp; Roboflow Datasets &nbsp;•&nbsp; Circular Electronics Research
          </p>
        </div>
      </footer>
    </div>
  );
}
