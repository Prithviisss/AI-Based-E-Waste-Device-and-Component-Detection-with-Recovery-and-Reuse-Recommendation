# Limitations

Reuse and recovery potential are determined by a knowledge base because no public dataset
provides these labels. Visible condition is limited to a low-confidence manual-inspection
flag. RAM and SSD detection is trained on a very small public dataset and is expected to be
weaker than PCB part detection. Public images are cleaner than real e-waste photos, so
real-world accuracy may be lower. Aluminium is inferred, not detected.

Future work: collect an annotated condition dataset and train XGBoost/Random Forest with
SHAP to replace the rules.

---

## Extended Notes

### Domain Gap

The classifier and detectors are trained on public internet images which tend to be clean,
well-lit studio-style photographs. Real e-waste photos (as shown in the `demo/` folder)
are typically dirtier, lower-resolution, and taken at unusual angles.
This gap means real-world metrics are expected to be lower than the test-set metrics
reported in `reports/metrics.json`.

### Dataset Size

| Module | Training samples | Risk |
|--------|-----------------|------|
| Classifier | ~2,500 (after dedupe) | Moderate — EfficientNet pretrained weights mitigate this |
| PCB detector | ✏ (confirm after download) | Moderate |
| RAM/SSD detector | ~150 (80 % of ~188) | High — small dataset; fallback may be needed |

### Recovery Logic

The recovery recommendation system is **not a machine learning model**. It applies explicit
if-else rules and a MILP optimizer over a manually curated knowledge base. Consequence:

- It cannot learn from new data without human updates to the knowledge base.
- All cited figures should be re-checked periodically as market prices and emission factors change.
- The MILP objective (maximise value or CO₂ saving) may not align with local regulations
  or the recycler's constraints; always consult a certified e-waste handler.

### What Is and Is Not Detected

| Signal | Status |
|--------|--------|
| Device type | ✅ Detected (12 classes) |
| PCB components | ✅ Detected (up to 21 classes) |
| RAM modules | ✅ Detected (weak; small training set) |
| SSD modules | ✅ Detected (weak; small training set) |
| Physical condition | ❌ Not detected — flagged as "needs manual inspection" only |
| Aluminium | ❌ Inferred from heatsink detection, not directly detected |
| PCB defects | ❌ Out of scope — no defect-detection module |
| Specific battery chemistry | ❌ Hazard flag is applied to all batteries uniformly |
