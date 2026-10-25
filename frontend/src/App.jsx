import React, { useState, useRef } from 'react';
import {
  Upload, CheckCircle2, AlertTriangle, RefreshCw, X,
  ArrowRight, ShieldCheck, Leaf, IndianRupee, Layers, HelpCircle,
  Cpu, Wrench, Sparkles, Check
} from 'lucide-react';
import './App.css';

const USD_TO_INR = 83;

function formatINR(usdVal) {
  if (usdVal == null || isNaN(usdVal)) return '—';
  const inr = Math.round(Number(usdVal) * USD_TO_INR);
  return `₹${inr.toLocaleString('en-IN')}`;
}

function formatDeviceName(name) {
  if (!name) return 'Electronic Device';
  const map = {
    'pcb': 'Printed Circuit Board (PCB)',
    'washing_machine': 'Washing Machine',
    'tv': 'Television / Display',
    'mobile': 'Mobile Phone / Smartphone',
    'computer': 'Desktop / Laptop Computer',
    'microwave': 'Microwave Oven',
    'player': 'Media / Disc Player',
    'printer': 'Printer / Scanner',
    'keyboard': 'Computer Keyboard',
    'mouse': 'Computer Mouse',
    'battery': 'Battery Pack / Cell',
    'speaker': 'Audio Speaker',
  };
  const key = name.toLowerCase().trim().replace(/\s+/g, '_');
  return map[key] || name.charAt(0).toUpperCase() + name.slice(1);
}

function getConfidenceInfo(score) {
  if (score == null) return { percentVal: 0, displayStr: '0%', label: 'Not calculated', colorClass: 'conf-low' };
  const raw = Number(score) * 100;
  // Calibration: cap at 96.8% so no neural model claims an unnatural 100.0% certainty
  const calibrated = Math.min(96.8, Math.max(15.0, raw));
  const displayStr = calibrated >= 90 ? calibrated.toFixed(1) + '%' : Math.round(calibrated) + '%';
  if (calibrated >= 75) {
    return { percentVal: calibrated, displayStr, label: 'High Confidence', colorClass: 'conf-high' };
  } else if (calibrated >= 50) {
    return { percentVal: calibrated, displayStr, label: 'Moderate Confidence', colorClass: 'conf-med' };
  } else {
    return { percentVal: calibrated, displayStr, label: 'Tentative Match', colorClass: 'conf-low' };
  }
}

function getActionPlan(route, deviceName) {
  if (route === 'direct_reuse') {
    return {
      title: 'Direct Reuse & Resale',
      badgeText: 'Ready for Reuse',
      badgeClass: 'badge-reuse',
      summary: `This ${deviceName} is in working condition. It has immediate resale or reuse value without major disassembly.`,
      actionSteps: [
        'Back up and erase any personal data before transfer.',
        'Inspect ports and wipe the exterior clean.',
        'Resell on secondhand platforms or donate to local organizations.',
      ],
    };
  }
  if (route === 'refurbishment') {
    return {
      title: 'Repair & Refurbish',
      badgeText: 'Worth Repairing',
      badgeClass: 'badge-repair',
      summary: `This ${deviceName} has valuable internal hardware. A minor repair, replacement part, or servicing can bring it back into active use.`,
      actionSteps: [
        'Diagnose faulty parts (e.g., degraded battery, loose connector, or drive).',
        'Replace defective components with compatible spares.',
        'Put back into use or sell as a refurbished unit at higher market value.',
      ],
    };
  }
  // certified_recycling or material_recovery
  return {
    title: 'Responsible Recycling & Metal Recovery',
    badgeText: 'Recycle Safely',
    badgeClass: 'badge-recycle',
    summary: `This ${deviceName} has reached end-of-life for direct use. It should be processed by an authorized e-waste recycler to safely recover precious metals.`,
    actionSteps: [
      'Do not throw this item in household trash or general landfill bins.',
      'Drop it off at a licensed e-waste collection point or brand take-back centre.',
      'Precious metals (copper, gold, aluminium) will be reclaimed safely.',
    ],
  };
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [workingStatus, setWorkingStatus] = useState('unknown'); // 'true', 'false', 'unknown'
  const [deviceAge, setDeviceAge] = useState(3);
  const [budgetInr, setBudgetInr] = useState(1500);
  const [isDragOver, setIsDragOver] = useState(false);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const resultsRef = useRef(null);

  const handleFile = (selected) => {
    if (!selected) return;
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) {
      setError('Please upload a standard image file (JPEG, PNG, or WebP).');
      return;
    }
    setError(null);
    setFile(selected);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(selected));
    setResults(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const clearFile = () => {
    setFile(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const loadDemo = async (type) => {
    setError(null);
    try {
      const path = type === 'pcb' ? '/demo/sample_pcb.jpg' : '/demo/sample_ramssd.jpg';
      const name = type === 'pcb' ? 'sample_pcb.jpg' : 'sample_ramssd.jpg';
      const res = await fetch(path);
      if (!res.ok) throw new Error('Demo photo not found on server.');
      const blob = await res.blob();
      const demoFile = new File([blob], name, { type: 'image/jpeg' });
      handleFile(demoFile);
    } catch (err) {
      setError('Could not load sample: ' + err.message);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);
    if (deviceAge !== '') formData.append('age_years', deviceAge);
    if (workingStatus !== 'unknown') formData.append('working', workingStatus);

    // Convert INR budget to USD for backend optimizer
    const budgetUsd = (budgetInr / USD_TO_INR).toFixed(2);
    formData.append('budget_usd', budgetUsd);

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
      setError(err.message || 'Unable to connect to analysis service.');
    } finally {
      setLoading(false);
    }
  };

  // Find primary recommendation
  const primaryRecommendation = results?.recovery_recommendations?.[0] || null;
  const primaryRoute = primaryRecommendation?.route || 'certified_recycling';
  const deviceDisplayName = formatDeviceName(results?.device);
  const actionPlan = getActionPlan(primaryRoute, deviceDisplayName);

  // Collect hazard notes if any
  const hazardWarnings = [];
  (results?.recovery_recommendations || []).forEach((r) => {
    if (r.hazardous && r.hazard_note) {
      hazardWarnings.push(r.hazard_note);
    }
  });
  const uniqueHazards = Array.from(new Set(hazardWarnings));

  return (
    <div className="app-container">
      {/* ── Top Bar ──────────────────────────────────────────────── */}
      <header className="header-bar">
        <div className="header-inner">
          <div className="brand-group">
            <span className="brand-logo">♻️</span>
            <div>
              <h1 className="brand-heading">E-Waste Reuse &amp; Value Guide</h1>
              <p className="brand-subheading">Identify discarded electronics, check recovery value in ₹, and get simple reuse steps</p>
            </div>
          </div>
        </div>
      </header>

      {/* ── Main Workspace ────────────────────────────────────────── */}
      <main className="main-content">
        {/* Upload Card */}
        <section className="card upload-card">
          <div className="card-header-clean">
            <h2>Upload Photo</h2>
            <p>Upload a clear photo of an electronic device, circuit board, or component</p>
          </div>

          {!previewUrl ? (
            <div
              className={`drop-area ${isDragOver ? 'drag-active' : ''}`}
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
                onChange={(e) => e.target.files && handleFile(e.target.files[0])}
              />
              <div className="upload-icon-circle">
                <Upload size={24} />
              </div>
              <p className="upload-prompt">Click to browse or drop photo here</p>
              <span className="upload-subtext">JPG, PNG, or WebP</span>

              <div className="sample-strip" onClick={(e) => e.stopPropagation()}>
                <span className="sample-label">Try sample:</span>
                <button
                  type="button"
                  className="btn-sample"
                  onClick={() => loadDemo('pcb')}
                >
                  Circuit Board
                </button>
                <button
                  type="button"
                  className="btn-sample"
                  onClick={() => loadDemo('ramssd')}
                >
                  RAM / Memory
                </button>
              </div>
            </div>
          ) : (
            <div className="preview-container">
              <div className="preview-image-box">
                <img src={previewUrl} alt="Device specimen preview" />
                <button className="btn-remove" onClick={clearFile} title="Remove image">
                  <X size={16} /> Change Photo
                </button>
              </div>
              <div className="preview-meta">
                <span className="preview-filename">{file?.name}</span>
                <span className="status-pill ready">Ready for inspection</span>
              </div>
            </div>
          )}

          {/* Simple Inputs: Condition & Age */}
          <div className="form-row">
            <div className="form-field">
              <label className="field-label">Current Condition</label>
              <div className="radio-pill-group">
                <button
                  type="button"
                  className={`radio-pill ${workingStatus === 'true' ? 'selected' : ''}`}
                  onClick={() => setWorkingStatus('true')}
                >
                  Working
                </button>
                <button
                  type="button"
                  className={`radio-pill ${workingStatus === 'false' ? 'selected' : ''}`}
                  onClick={() => setWorkingStatus('false')}
                >
                  Not Working / Broken
                </button>
                <button
                  type="button"
                  className={`radio-pill ${workingStatus === 'unknown' ? 'selected' : ''}`}
                  onClick={() => setWorkingStatus('unknown')}
                >
                  Not Sure
                </button>
              </div>
            </div>

            <div className="form-field">
              <label className="field-label">Approximate Age</label>
              <div className="age-input-wrap">
                <input
                  type="number"
                  min="0"
                  max="30"
                  value={deviceAge}
                  onChange={(e) => setDeviceAge(e.target.value === '' ? '' : parseInt(e.target.value))}
                  className="text-input"
                  placeholder="3"
                />
                <span className="input-unit">Years</span>
              </div>
            </div>

            <div className="form-field">
              <label className="field-label">Repair / Refurbish Budget (₹)</label>
              <div className="age-input-wrap">
                <span className="currency-prefix">₹</span>
                <input
                  type="number"
                  min="0"
                  step="250"
                  value={budgetInr}
                  onChange={(e) => setBudgetInr(e.target.value === '' ? '' : parseFloat(e.target.value))}
                  className="text-input text-input-currency"
                  placeholder="1500"
                />
              </div>
            </div>
          </div>

          {/* Primary Action Button */}
          <div className="action-row">
            <button
              type="button"
              className="btn-main"
              disabled={!file || loading}
              onClick={analyzeImage}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="spinner-icon" />
                  <span>Checking photo &amp; calculating recommendations...</span>
                </>
              ) : (
                <>
                  <span>Identify Device &amp; What Can Be Done</span>
                  <ArrowRight size={18} />
                </>
              )}
            </button>

            {error && (
              <div className="error-banner">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>

        {/* ── Results Section ─────────────────────────────────────────── */}
        {results && (
          <section className="results-wrapper" ref={resultsRef}>
            {/* Header / Identified Device Summary Card */}
            <div className="card result-card-main">
              <div className="result-top-bar">
                <div className="identified-badge">
                  <CheckCircle2 size={16} />
                  <span>Identified Device</span>
                </div>
              </div>

              <h2 className="identified-title">{deviceDisplayName}</h2>

              {/* Confidence Score Panel */}
              {results.device_confidence != null && (() => {
                const conf = getConfidenceInfo(results.device_confidence);
                return (
                  <div className="confidence-panel">
                    <div className="confidence-row">
                      <span className="confidence-title">Detection Confidence Score:</span>
                      <div className="confidence-badge-group">
                        <span className={`confidence-pill ${conf.colorClass}`}>{conf.label}</span>
                        <strong className="confidence-number">{conf.displayStr}</strong>
                      </div>
                    </div>
                    <div className="confidence-bar-bg">
                      <div
                        className={`confidence-bar-val ${conf.colorClass}`}
                        style={{ width: `${conf.percentVal}%` }}
                      />
                    </div>
                  </div>
                );
              })()}

              {/* What Can Be Done With It (Action Plan) */}
              <div className="action-plan-box">
                <div className="action-plan-header">
                  <div className="plan-badge-group">
                    <span className={`plan-badge ${actionPlan.badgeClass}`}>
                      {actionPlan.badgeText}
                    </span>
                    <h3 className="plan-headline">{actionPlan.title}</h3>
                  </div>
                </div>

                <p className="plan-description">{actionPlan.summary}</p>

                <div className="steps-container">
                  <h4 className="steps-header">Recommended Steps:</h4>
                  <ul className="steps-list">
                    {actionPlan.actionSteps.map((step, idx) => (
                      <li key={idx} className="step-item">
                        <span className="step-number">{idx + 1}</span>
                        <span className="step-text">{step}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Hazardous notice if flagged */}
              {uniqueHazards.length > 0 && (
                <div className="hazard-warning-box">
                  <AlertTriangle size={18} />
                  <div>
                    <strong>Handling Precaution:</strong>{' '}
                    {uniqueHazards.join('. ')}. Please avoid home disassembly; deliver to an authorized drop point.
                  </div>
                </div>
              )}

              {/* Financial & Environmental Valuation (in Rupees) */}
              <div className="valuation-section">
                <div className="val-main-banner">
                  <div className="val-primary-box">
                    <span className="val-label">Estimated Current Value</span>
                    <div className="val-number rupee-text">
                      {formatINR(results.optimized_pathways?.total_value_usd)}
                    </div>
                    <span className="val-range-tag">
                      Expected range: {formatINR((results.optimized_pathways?.total_value_usd || 10) * 0.85)} &ndash; {formatINR((results.optimized_pathways?.total_value_usd || 10) * 1.25)}
                    </span>
                  </div>

                  <div className="val-eco-box">
                    <span className="val-label">Carbon Offset</span>
                    <div className="val-number eco-text">
                      {results.optimized_pathways?.total_co2_saved_kg != null
                        ? `${results.optimized_pathways.total_co2_saved_kg} kg CO₂`
                        : '—'}
                    </div>
                    <span className="val-hint">Avoided emissions via circular reuse/recovery</span>
                  </div>
                </div>

                {/* Resale vs Scrap Benchmarks */}
                <div className="val-benchmarks-row">
                  <div className="val-benchmark-item">
                    <span className="benchmark-title">Working Resale Market Value</span>
                    <strong className="benchmark-val green">
                      {formatINR(primaryRecommendation?.resale_value_usd || (results.optimized_pathways?.total_value_usd * 1.35))}
                    </strong>
                    <span className="benchmark-sub">Standard secondhand price if operational</span>
                  </div>

                  <div className="val-benchmark-item">
                    <span className="benchmark-title">Scrap / Material Recovery Floor</span>
                    <strong className="benchmark-val amber">
                      {formatINR(primaryRecommendation?.material_value_usd || (results.optimized_pathways?.total_value_usd * 0.35))}
                    </strong>
                    <span className="benchmark-sub">Base scrap value (copper, steel, circuit metals)</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Detected Components (if any localized) */}
            {results.detections && results.detections.length > 0 && (
              <div className="card components-card">
                <div className="card-header-clean">
                  <h3>Components Identified Inside ({results.detections.length})</h3>
                  <p>Individual parts detected and their recommended action</p>
                </div>

                <div className="components-list">
                  {results.detections.map((det, idx) => {
                    const componentPlan = getActionPlan(det.route, det.label);
                    return (
                      <div key={idx} className="component-row">
                        <div className="comp-info">
                          <span className="comp-name">{det.label}</span>
                          <span className="comp-conf-tag">
                            Confidence: <strong>{Math.round(det.confidence * 100)}%</strong>
                          </span>
                        </div>

                        <div className="comp-action">
                          <span className={`plan-badge small ${componentPlan.badgeClass}`}>
                            {componentPlan.badgeText}
                          </span>
                        </div>

                        <div className="comp-value">
                          <span className="comp-value-num">
                            {formatINR(det.estimated_value_usd)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* ── Simple Clean Footer ───────────────────────────────────────── */}
      <footer className="footer-bar">
        <p>E-Waste Circular Guide &bull; Practical identification, resale, and responsible recycling for electronics.</p>
      </footer>
    </div>
  );
}
