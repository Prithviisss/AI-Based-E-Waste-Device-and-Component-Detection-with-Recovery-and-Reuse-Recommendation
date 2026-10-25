/* web/app.js — E-Waste RecoveryAI Frontend */
"use strict";

const API_BASE = "";  // same origin

// ── DOM refs ────────────────────────────────────────────────────────────────
const dropzone      = document.getElementById("dropzone");
const fileInput     = document.getElementById("file-input");
const previewWrap   = document.getElementById("preview-wrap");
const previewImg    = document.getElementById("preview-img");
const removeImg     = document.getElementById("remove-img");
const analyzeBtn    = document.getElementById("analyze-btn");
const btnLabel      = document.getElementById("btn-label");
const spinner       = document.getElementById("spinner");
const resultsSection = document.getElementById("results-section");
const resultsInner  = document.getElementById("results-inner");

let selectedFile = null;

// ── File handling ────────────────────────────────────────────────────────────
function setFile(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Only JPEG, PNG, or WebP images are accepted.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.classList.remove("hidden");
  dropzone.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function clearFile() {
  selectedFile = null;
  previewImg.src = "";
  previewWrap.classList.add("hidden");
  dropzone.classList.remove("hidden");
  analyzeBtn.disabled = true;
  resultsSection.classList.add("hidden");
}

fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
removeImg.addEventListener("click", clearFile);

// Drag-and-drop
dropzone.addEventListener("dragover", e => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
});
dropzone.addEventListener("click", () => fileInput.click());

// ── Submit ───────────────────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  setLoading(true);
  resultsSection.classList.add("hidden");

  const fd = new FormData();
  fd.append("image", selectedFile);

  const age = document.getElementById("age-input").value;
  if (age) fd.append("age_years", age);

  const working = document.getElementById("working-select").value;
  if (working) fd.append("working", working);

  const budget = document.getElementById("budget-input").value || "20";
  fd.append("budget_usd", budget);

  try {
    const resp = await fetch(`${API_BASE}/analyze`, { method: "POST", body: fd });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || "Server error");
    }
    const data = await resp.json();
    renderResults(data);
    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

// ── Render results ───────────────────────────────────────────────────────────
function renderResults(d) {
  const confPct = (d.device_confidence * 100).toFixed(1);
  const routeBadge = (route) => {
    if (!route) return "—";
    const cls = route === "direct_reuse" ? "badge-green"
              : route === "refurbishment" ? "badge-yellow" : "badge-red";
    return `<span class="badge ${cls}">${route.replace(/_/g," ")}</span>`;
  };

  let html = `
    <!-- Device card -->
    <div class="result-grid">
      <div class="result-card">
        <h3>Identified Device</h3>
        <div class="result-value">${capitalize(d.device)}</div>
        <div class="conf-bar"><div class="conf-fill" style="width:${confPct}%"></div></div>
        <p style="font-size:0.8rem;color:var(--muted);margin-top:0.4rem">${confPct}% confidence</p>
      </div>
      <div class="result-card">
        <h3>Optimized Pathway</h3>
        <div style="font-size:1.1rem;font-weight:700;color:${d.optimized_pathways?.status==='optimal'?'var(--primary)':'var(--danger)'}">
          ${d.optimized_pathways?.status?.toUpperCase() || '—'}
        </div>
        ${d.optimized_pathways?.status === 'optimal' ? `
          <p style="font-size:0.85rem;margin-top:0.5rem">
            💰 ₹${Math.round((d.optimized_pathways.total_value_usd || 0) * 83).toLocaleString('en-IN')} recovered &nbsp;·&nbsp;
            🌱 ${d.optimized_pathways.total_co2_saved_kg} kg CO₂ saved
          </p>
          <p style="font-size:0.78rem;color:var(--muted)">Selected: ${(d.optimized_pathways.selected||[]).join(', ')||'—'}</p>
        ` : `<p style="font-size:0.85rem;color:var(--danger);margin-top:0.5rem">${d.optimized_pathways?.message||''}</p>`}
      </div>
    </div>`;

  // Detections table
  if (d.detections?.length) {
    html += `
    <div class="result-card" style="margin-bottom:1rem">
      <h3 style="font-size:1rem;font-weight:600;margin-bottom:0.75rem">🔍 Detected Components</h3>
      <div class="result-table-wrap">
        <table>
          <thead><tr>
            <th>Component</th><th>Confidence</th><th>Action</th>
            <th>Value (INR)</th><th>CO₂ Saved</th>
          </tr></thead>
          <tbody>
            ${d.detections.map(det => `
              <tr>
                <td>${det.label}</td>
                <td>${(det.confidence*100).toFixed(1)}%</td>
                <td>${routeBadge(det.route)}</td>
                <td>${det.estimated_value_usd != null ? '₹' + Math.round(det.estimated_value_usd * 83).toLocaleString('en-IN') : '—'}</td>
                <td>${det.co2_saved_kg != null ? det.co2_saved_kg+' kg' : '—'}</td>
              </tr>`).join('')}
          </tbody>
        </table>
      </div>
    </div>`;
  }

  // Unmapped warning
  if (d.unmapped?.length) {
    html += `<div class="warn-box">⚠ Unmapped components (see class_map.yaml): ${d.unmapped.join(', ')}</div>`;
  }

  // Flags
  const allFlags = [];
  (d.recovery_recommendations || []).forEach(r => (r.flags||[]).forEach(f => allFlags.push(f)));
  const uniqueFlags = [...new Set(allFlags)];
  if (uniqueFlags.includes("needs_manual_inspection")) {
    html += `<div class="warn-box" style="margin-top:0.75rem">⚠ Low-confidence detection(s) flagged for <strong>manual inspection</strong>.</div>`;
  }

  // Heatmaps
  if (d.heatmap_urls?.length) {
    html += `
    <div class="result-card" style="margin-top:1rem">
      <h3 style="font-size:1rem;font-weight:600;margin-bottom:0.75rem">🔬 Grad-CAM Explainability</h3>
      <div class="heatmap-grid">
        ${d.heatmap_urls.map(h => `
          <div class="heatmap-item">
            <img src="${h.url}" alt="${h.label}">
            <div class="heatmap-label">${h.label}</div>
          </div>`).join('')}
      </div>
    </div>`;
  }

  // Report link
  if (d.report_url) {
    html += `
    <div style="text-align:center;margin-top:1.5rem">
      <a href="${d.report_url}" target="_blank" class="btn-primary">
        📄 Open Full Report
      </a>
    </div>`;
  }

  resultsInner.innerHTML = html;
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function setLoading(on) {
  analyzeBtn.disabled = on;
  btnLabel.textContent = on ? "Analyzing…" : "Analyze Device";
  spinner.classList.toggle("hidden", !on);
}

function showError(msg) {
  resultsSection.classList.remove("hidden");
  resultsInner.innerHTML = `<div class="error-box">❌ <strong>Error:</strong> ${escapeHtml(msg)}</div>`;
  resultsSection.scrollIntoView({ behavior: "smooth" });
}

function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
function escapeHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}
