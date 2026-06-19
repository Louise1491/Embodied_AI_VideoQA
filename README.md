# Embodied AI Training Data — Video QA Toolkit

A methodology and toolset for evaluating egocentric video data quality for embodied AI / robot manipulation training.

---

## Why This Exists

Most video QA tools focus on broadcast quality metrics (bitrate, compression artifacts). Most embodied AI repos focus on training pipelines. This toolkit addresses the gap between them: **evaluating whether a video clip meets the specific requirements for robot manipulation training data collection.**

Training embodied AI models requires large-scale egocentric video of manipulation tasks. But **video volume ≠ training value** — a clip that looks fine to a human reviewer can be systematically unusable for model training due to issues that are invisible without a structured evaluation framework.

This repo documents:
1. A **4-layer QA framework** for evaluating egocentric video data quality
2. A **7-pattern checklist** for auditing data collection guidelines themselves
3. **Automation scripts** for Layer A (technical parameters) and Layer B (visual quality)

---

## The 4-Layer QA Framework

| Layer | What It Checks | Automation Level |
|-------|---------------|-----------------|
| **A — Technical Parameters** | Resolution, frame rate, bitrate, duration, codec | Fully automated |
| **B — Visual Quality** | Motion blur, hand visibility, focus/exposure | Semi-automated |
| **C — Content Compliance** | Scene type, PII detection | Partially automated |
| **D — Data Value Density** | Action primitive coverage, context diversity, long-tail ratio | Human judgment (framework in `/docs`) |

> The key insight: quality issues caught at Layer A cost near zero to detect. Issues that slip through to Layer D cost the most — both to detect and to remediate. The framework is designed to maximize early-stage catch rate.

---

## Scripts

### `scripts/technical_check.py`
Automated Layer A filter using ffprobe. Checks resolution, frame rate, bitrate, duration, and codec against configurable thresholds. Returns pass/fail per dimension with actionable output.

```bash
python scripts/technical_check.py --input ./videos --config config.yaml
```

### `scripts/motion_blur_detect.py`
Layer B visual quality detection. Current implementation uses Laplacian variance (v1). Known limitation: high-texture backgrounds inflate scores, causing false negatives in industrial scenes.

**Planned improvements:**
- **v2:** ROI-based detection — isolate hand region via MediaPipe Hand Landmark before computing sharpness score, eliminating background texture interference
- **v3:** Optical flow analysis — measure motion velocity between frames rather than static sharpness, enabling distinction between hand-motion blur (operator issue) and tool-vibration blur (not operator issue)

```bash
python scripts/motion_blur_detect.py --input ./videos --output ./reports
```

---

## Guideline Audit Framework

One of the less obvious skills in data QA is evaluating whether the **collection standard itself** is complete and internally consistent. A flawed guideline produces inconsistent QA decisions at scale — different reviewers reach different conclusions on identical clips.

Seven common defect patterns to check in any data collection guideline:

| Pattern | Description |
|---------|-------------|
| **Missing threshold** | Qualitative language ("avoid heavy compression") with no numeric floor |
| **Internal contradiction** | Two rules in the same document that conflict (e.g. "minimum 30s" vs "target 60s") |
| **Term ambiguity** | Key terms with multiple valid interpretations (e.g. "acting hand" vs "both hands") |
| **Unclear attribution** | Same observable issue can have multiple root causes; guideline doesn't distinguish |
| **Scenario gap** | Emerging or boundary scenario types not listed in allowed/disallowed lists |
| **Category gap** | A risk category (e.g. PII) covered partially, with common cases omitted |
| **Unit undefined** | Vague quantifiers ("small steps", "brief pause") without defined units or thresholds |

> Applying this checklist before executing QA at scale surfaces ambiguities that would otherwise produce inconsistent decisions across reviewers.

---

## Full Methodology

Detailed methodology documentation is in [`/docs/methodology.md`](docs/methodology.md), covering:

- Layer B technical deep-dive (ROI detection, optical flow analysis)
- Layer D: Data Value Density framework (action primitive coverage, context diversity index, long-tail coverage ratio)
- Guideline audit framework with examples for each pattern

---

## Background

This framework was developed through hands-on work evaluating egocentric video datasets for robot manipulation training, combined with research into embodied AI data infrastructure challenges. It draws on prior experience in automotive data pipeline design and cross-functional quality management.

The methodology is domain-agnostic at the framework level — the 4-layer structure and guideline audit checklist apply to any egocentric video collection standard, not just robot manipulation.

---

## Status

| Component | Status |
|-----------|--------|
| 4-layer framework | ✅ Complete |
| `technical_check.py` (Layer A) | ✅ Working |
| `motion_blur_detect.py` v1 (Layer B) | ✅ Working (known limitations) |
| `motion_blur_detect.py` v2 — ROI-based | 🔧 In progress |
| `motion_blur_detect.py` v3 — Optical flow | 📐 Designed, pending implementation |
| Guideline audit framework | ✅ Complete |
| Layer D value density framework | 📄 Documented in `/docs` |

---

## Contact

Louise Wang · [LinkedIn](https://www.linkedin.com/in/louiseluyingwang) · USC Marshall MSBA
