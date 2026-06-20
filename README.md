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
python technical_check.py <video_folder>
```

### `scripts/motion_blur_detect.py` (v1)
Layer B visual quality detection — full-frame Laplacian variance implementation.

Known limitation: high-texture backgrounds (metal grids, wood grain, patterned fabric) inflate the per-frame variance score, masking blur in the hand region. Use as a rough first-pass filter only.

```bash
python motion_blur_detect.py <video_file>
python motion_blur_detect.py <video_folder>    # batch mode
```

### `scripts/motion_blur_detect_v2.py` (v2 — recommended)
Layer B visual quality detection — ROI-based implementation.

Instead of scoring the full frame, v2 first localises the hand region via HSV skin colour segmentation, then computes Laplacian variance only within that bounding box. Background texture no longer affects the score. Also flags frames where no hand is detected as `HAND_NOT_VISIBLE` — a distinct quality issue from blur.

```bash
python motion_blur_detect_v2.py <video_file>
python motion_blur_detect_v2.py <video_folder>    # batch mode
```

**Validated on a set of egocentric manipulation clips across varied industrial scene types. Real-world accuracy by scene type:**

| Scene type | V2 reliability | Notes |
|---|---|---|
| Dark skin + light/neutral background | ✅ Reliable | Core use case |
| Full-frame camera shake | ⚠️ May miss | Planned: V3 optical flow |
| Dark skin + dark background | ⚠️ Over-reports | Low contrast and blur produce similar Laplacian scores |
| Light skin + light background | ❌ ROI mislocation | Skin segmentation picks up background instead of hand |
| High-texture background | ❌ May miss blur | Background patterns interfere with ROI detection |

**V1 vs V2 — when to use which:**
- Use **v2** when operators have darker skin tones working on light or neutral surfaces
- Use **v1** as a rough first-pass for all other scene types until v3 is available
- Both versions produce a CSV report and a sharpness curve chart (PNG)

**Planned v3:** Optical flow analysis — measures motion velocity between frames rather than static sharpness. Will be colour-agnostic and will distinguish hand-motion blur (operator issue) from tool-vibration blur (not operator fault).

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
| `motion_blur_detect.py` v1 (Layer B) | ✅ Working (known limitations documented) |
| `motion_blur_detect_v2.py` v2 (Layer B) | ✅ Working (validated across varied scene types) |
| `motion_blur_detect` v3 — Optical flow | 📐 Designed, pending implementation |
| Guideline audit framework | ✅ Complete |
| Layer D value density framework | 📄 Documented in `/docs` |

---

## Contact

Louise Wang · [LinkedIn](https://linkedin.com/in/louiseluyingwang/) · USC Marshall MSBA
