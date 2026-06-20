# Methodology: Egocentric Video QA for Robot Manipulation Training Data

This document covers the technical reasoning behind the 4-layer QA framework and the tools in this repository. It is intended for practitioners who want to understand not just *what* the scripts do, but *why* they are designed the way they are.

---

## Table of Contents

1. [Layer B Deep-Dive: Visual Quality Assessment](#layer-b-deep-dive)
2. [Layer D: Data Value Density Framework](#layer-d-data-value-density)
3. [Guideline Audit Framework: Detailed Examples](#guideline-audit-framework)

---

## Layer B Deep-Dive

### Why Full-Frame Sharpness Scoring Fails in Industrial Scenes

The standard approach to motion blur detection — Laplacian variance across the full frame — was designed for general photography. It measures the density of edges in an image: more edges = sharper image.

This assumption breaks down in egocentric industrial video for one core reason: **the background is often more textured than the hands**.

A metal grid, wood-grain workbench, or patterned fabric in the background contributes far more edges to the full-frame score than the relatively smooth skin of an operator's hand. The result is a systematic false negative: a frame where the hand is severely blurred but the background is sharp scores as "acceptable."

This is not a corner case — it is the default condition in most industrial capture environments.

### The V2 Fix: ROI-Based Detection

The solution is to compute sharpness only where it matters — the hand region.

**Step 1: Localise the hand**

V2 uses HSV colour segmentation to detect skin-coloured regions in each frame. Two HSV ranges are combined to cover a broad spectrum of skin tones:

```
Range 1: H[0–20],   S[20–255], V[70–255]   (lighter skin tones)
Range 2: H[160–180], S[20–255], V[70–255]  (wraps around hue wheel)
```

Morphological operations (closing, opening) clean up noise and fill small gaps in the detected region. The largest qualifying contour is selected as the hand bounding box.

**Step 2: Compute sharpness within the ROI**

Laplacian variance is computed only within the padded bounding box. The background — regardless of how textured it is — has no effect on the score.

**Step 3: Flag frames where no hand is detected**

If no skin region above the minimum area threshold is found, the frame is flagged as `HAND_NOT_VISIBLE` rather than silently scored. This distinguishes two separate quality issues that V1 conflates:
- Hand is in frame but blurry → `BLUR` / `SEVERE_BLUR`
- Hand has left the frame entirely → `HAND_NOT_VISIBLE`

### V2 Validation Results

Tested across egocentric manipulation clips spanning varied scene types and operator demographics. Key findings:

**Where V2 works well:**
- Dark skin tones on light or neutral backgrounds (e.g. stainless steel sink, white workbench)
- Sustained motion blur throughout a clip is reliably detected
- V2 ROI scores on known-blurry clips were 3–4× lower than V1 full-frame scores on identical frames

**Where V2 has known limitations:**

| Failure mode | Root cause | Planned fix |
|---|---|---|
| Full-frame camera shake | Camera and hand move together; ROI sharpness unaffected | V3 optical flow |
| Dark skin + dark background | Low contrast and blur produce similar Laplacian scores | Cannot be resolved by sharpness alone; requires lighting standards |
| Light skin + light background | Skin segmentation misidentifies background as hand | V3 optical flow (colour-agnostic) |
| High-texture backgrounds near skin colour | ROI bounding box includes background texture | V3 optical flow |

### Planned V3: Optical Flow Analysis

Optical flow measures *pixel movement between frames* rather than static sharpness within a frame. This gives V3 two advantages over V2:

**1. Colour-agnostic:** Optical flow operates on motion vectors, not colour. It works equally well regardless of skin tone or background colour.

**2. Distinguishes blur sources:** By computing motion separately for the hand region and the background, V3 can distinguish:
- **Hand-motion blur** — the operator's hand moved too fast. This is an operator execution issue; the clip should be flagged and the operator given specific feedback.
- **Tool-vibration blur** — the tool itself vibrates at high RPM (e.g. an orbital sander) while the operator's hand remains relatively still. The hand keypoints are still recoverable; this should not be flagged as a speed violation.

V1 and V2 cannot make this distinction. Misclassifying tool-vibration blur as operator error generates incorrect feedback that cannot be acted on.

---

## Layer D: Data Value Density Framework

### Why Hours Are a Poor Proxy for Training Value

Industry convention measures dataset size in hours. This metric is intuitive but misleading: the marginal training value of additional data drops sharply once a particular combination of task, operator, environment, and difficulty has been sufficiently covered.

A dataset of 1,000 hours where the same operator performs the same task in the same environment at the same difficulty level may provide less training signal than 200 hours of well-designed varied data.

The framework below operationalises "data value density" as three measurable dimensions.

### Dimension 1: Action Primitive Coverage

**Definition:** The proportion of target manipulation primitives covered at sufficient sample depth in the current dataset.

Core manipulation primitives for general-purpose robot training:

| Primitive | Description |
|---|---|
| Grasp | Fingers close around an object |
| Place | Object released at a target location |
| Insert | Object placed into a constrained aperture |
| Rotate | Object or fastener turned about an axis |
| Sort | Object categorised and moved to a corresponding location |
| Fold | Flexible material creased along a line |
| Pull | Object drawn toward the operator |

**Target:** Each primitive should have a minimum sample depth (e.g. ≥500 clips) before marginal returns diminish significantly. Coverage = primitives at target depth / total target primitives.

### Dimension 2: Context Diversity Index

**Definition:** For a given action primitive, how many distinct combinations of contextual variables have been captured.

Key contextual variables:

- **Operator demographics:** hand size, skin tone, dominant hand, grip style
- **Lighting:** overhead, side, low-light, mixed
- **Work surface:** colour, material, texture, height
- **Object properties:** size, weight, colour, material, surface texture

**Target:** Each primitive should be represented across at least 3 distinct combinations of these variables. Single-context repetition beyond a saturation point adds minimal training value.

### Dimension 3: Long-Tail Coverage Ratio

**Definition:** The proportion of clips that capture boundary conditions and failure-recovery scenarios rather than nominal execution.

Long-tail scenarios include:
- Object slightly misaligned or tilted at pickup
- Grasp point suboptimal; requires mid-task adjustment
- Low-light conditions
- Non-dominant hand operation
- Interruption and recovery mid-task

**Target:** Approximately 20% of clips should be long-tail scenarios. Models trained exclusively on nominal execution tend to fail precisely in these boundary conditions — which are also the cases most likely to be encountered in real deployment.

### Practical Application

This framework informs two decisions:

**1. Task card design:** Each collection task card should specify which primitive it targets, what contextual variation is required for this session, and whether it is a nominal or long-tail variant. This prevents suppliers from defaulting to the path of least resistance (repeating the same task in the same way).

**2. Collection prioritisation:** Before commissioning new data, audit the current dataset against the three dimensions. Commission what is missing, not what is easiest to produce.

---

## Guideline Audit Framework

### Why Guidelines Need Auditing

A data collection guideline is itself a form of code: it specifies expected behaviour and defines what constitutes a violation. Like code, it can contain bugs.

A flawed guideline is more damaging than no guideline, because it creates an illusion of consistency while actually producing systematically incorrect QA decisions. Reviewers following a contradictory guideline will make different calls on identical clips, introducing noise into the dataset at the source.

The seven patterns below represent the most common failure modes observed across data collection standards. Each can be checked in under five minutes per clause.

### Pattern 1: Missing Threshold

**What it looks like:** Qualitative language where a numeric value is needed.

> *"Use high-bitrate H.264. Avoid heavy compression."*

**Why it fails:** "Heavy" is undefined. One reviewer may flag 8,000 kbps; another may accept it. At scale, this inconsistency means the same supplier submitting the same quality of footage will pass QA with some reviewers and fail with others.

**Fix template:** Replace qualitative language with a two-tier numeric standard. Example:

> *"Minimum bitrate: 12,000 kbps for 1920×1080 at 30fps. Files below 8,000 kbps will be rejected. Files between 8,000–12,000 kbps will be flagged as Major."*

---

### Pattern 2: Internal Contradiction

**What it looks like:** Two clauses in the same document that cannot both be satisfied simultaneously.

> *"Each episode must be at least 30 seconds."*
> *"Episode length should match or exceed [reference dataset] per-episode length (~61 seconds)."*

**Why it fails:** A 45-second clip satisfies the first clause and violates the second. There is no principled basis for a decision. The reviewer must choose which clause to apply — and different reviewers will choose differently.

**Fix template:** Clarify the logical relationship between the two standards.

> *"Hard minimum: 30 seconds (absolute reject floor). Target: 60 seconds (aligned with reference dataset). Clips between 30–60 seconds should be flagged as Major and re-shot where operationally feasible."*

---

### Pattern 3: Term Ambiguity

**What it looks like:** A key term used inconsistently within the same document.

> *"The operator's acting hand(s) must be visible in every frame."*
> *"Before each take, tilt the head down until both hands sit in the lower third."*

**Why it fails:** "Acting hand(s)" implies only the hand actively performing the manipulation is required to be in frame. "Both hands" implies all hands must always be visible. A clip where one hand is idle and temporarily off-screen is compliant under the first reading and non-compliant under the second.

**Fix template:** Define the term precisely and apply it consistently.

> *"The acting hand — the hand directly performing the current sub-task — must be in frame at all times. A non-acting hand (idle or transitioning between sub-tasks) may leave the frame for no more than 3 consecutive seconds."*

---

### Pattern 4: Unclear Attribution

**What it looks like:** A clause that flags a symptom without distinguishing between multiple possible causes, each of which requires different corrective action.

> *"Slow, deliberate movements. No flicking or fast motions."*

**Why it fails:** Motion blur can result from the operator's hand moving too fast, or from a high-RPM tool vibrating while the hand is stationary. These look similar in the output but have completely different causes and different fixes. Treating them identically gives operators incorrect feedback: telling someone to "move slower" when the blur is caused by a sander vibrating at 12,000 RPM is not actionable.

**Fix template:** Distinguish causes explicitly.

> *"Tool-induced blur — caused by the tool's own vibration rather than hand movement — is not grounds for rejection, provided the operator's hand keypoints remain distinguishable frame-to-frame. Only blur caused by hand movement itself (where the hand's trajectory causes keypoint loss) triggers a speed violation."*

---

### Pattern 5: Scenario Gap

**What it looks like:** The allowed/disallowed scenario list was written for one context and has not been updated as new use cases emerged.

> *Allowed: factory floor, assembly line, workshop, warehouse, indoor lab.*

**Why it fails:** This list does not address service-industry environments — restaurant prep stations, hotel housekeeping, retail stockrooms. These environments have manipulation-focused, repetitive tasks with workstation characteristics similar to light manufacturing. A QA reviewer cannot determine whether to accept or reject footage from these environments based on the existing guideline.

**Fix template:** Add an explicit ruling on the unlisted category.

> *"Service-industry environments (restaurant prep stations, hotel housekeeping areas, retail stockrooms) are [in scope / out of scope]. If in scope, the capture area must include a fixed counter or workbench with defined workstation boundaries."*

---

### Pattern 6: Category Gap

**What it looks like:** A risk category is listed but defined by enumeration rather than by principle, leaving obvious members of the category unaddressed.

> *"No PII: faces, license plates, ID cards, screens, addresses must not appear."*

**Why it fails:** This list covers the most common PII forms but omits others. A wrist tattoo containing a name or date is functionally equivalent to a name tag as a PII risk, but is not mentioned. A reviewer has no basis for a decision and must either guess or escalate every instance.

**Fix template:** Define the category by principle, then enumerate examples.

> *"No personally identifiable information (PII) may appear in the footage. PII includes any information that could identify a specific individual, including but not limited to: faces, license plates, ID cards, screens displaying personal data, written names or addresses, and tattoos containing names, dates, or other identifying text. Tattoos must be covered with clothing or a bandage before capture."*

---

### Pattern 7: Unit Undefined

**What it looks like:** A quantitative constraint expressed with a vague quantifier rather than a measurable unit.

> *"Navigation that is part of manipulation (e.g. moving along the workbench) is acceptable. At most 1–2 small steps between sub-tasks."*

**Why it fails:** "Small steps" is not defined. A step taken by a 160cm operator is physically different from one taken by a 190cm operator. More importantly, the clause does not define what constitutes "between sub-tasks" versus "between workstations" — the distinction the clause is trying to draw.

**Fix template:** Replace the vague quantifier with a measurable unit and clarify the boundary condition.

> *"Sub-task transitions involving movement must not exceed 1 metre / 3 steps, and must remain within a single functional zone (e.g. a single workbench or prep counter). Movement between separate functional zones (e.g. from a heating station to a storage area) is not considered part of the manipulation and constitutes a locomotion violation."*

---

*Last updated: May 2026*
