---
layout: post
title: "From 72% to 91% Accuracy: Engineering LLaMA for Document Extraction"
date: 2026-04-21
categories: llm nlp systems
---

## 🚨 Problem Statement

Extracting structured key-value data from OCR documents is **not just NLP** — it's a **layout + reasoning problem**.

Raw OCR output:
- Noisy
- Broken structure
- Lost spatial context

Traditional regex fails. Vanilla LLM prompts are unstable.

---

## 🏗️ System Architecture

```
        +-------------------+
        |   Input Image     |
        +-------------------+
                  |
                  v
        +-------------------+
        |   OCR Engine      |
        | (with bbox data)  |
        +-------------------+
                  |
                  v
        +-----------------------------+
        | Layout-Preserved Text Builder|
        +-----------------------------+
                  |
                  v
        +-------------------+
        |   LLaMA 3.1 8B    |
        | Prompt Engineered |
        +-------------------+
                  |
                  v
        +-------------------+
        | Post Processing   |
        | (bbox mapping)    |
        +-------------------+
```

---

## 🧠 Key Idea: Layout > Raw Text

Instead of feeding plain text:

❌ Bad:
```
Name: John Doe Address: NY
```

✅ Good:
```
[Name]        John Doe
[Address]     NY
```

Preserving layout improves semantic grouping.

---

## ⚙️ Implementation

### Step 1: Build Layout Text

```python

def build_layout_text(ocr_blocks):
    lines = sorted(ocr_blocks, key=lambda x: (x['y'], x['x']))
    result = []
    for block in lines:
        result.append(f"[{block['label']}] {block['text']}")
    return "\n".join(result)
```

---

### Step 2: Prompt Engineering

```python
prompt = f"""
Extract key-value pairs from the document.

Rules:
- Return JSON
- Do not hallucinate
- Use exact values

Document:
{layout_text}
"""
```

---

### Step 3: Post-processing with Bounding Boxes

```python

def map_to_bbox(extracted_keys, ocr_data):
    mapping = {}
    for key, value in extracted_keys.items():
        for block in ocr_data:
            if value in block['text']:
                mapping[key] = block['bbox']
    return mapping
```

---

## 📈 Results

| Approach              | Accuracy |
|----------------------|----------|
| Plain OCR + Regex    | 52%      |
| Raw LLM Prompt       | 72%      |
| Layout + Prompt      | 86%      |
| + Post-processing    | **91%**  |

---

## ⚠️ Challenges

- Multi-line values
- Table structures
- OCR noise

---

## 💡 Key Insights

- Prompting is **programming**
- Layout is **signal, not noise**
- Hybrid systems win in production

---

## 🚀 Final Thoughts

LLMs alone are not enough.

The real power comes from:
> Combining **CV + NLP + System Design**

This is where applied ML engineers stand out.
