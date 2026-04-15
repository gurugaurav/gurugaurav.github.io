---
layout: post
title: "Improving LLaMA for Document Key-Value Extraction"
date: 2026-04-16
categories: llm nlp
---

## Problem

Extracting structured data from OCR text is noisy and layout-sensitive.

## Approach

- Use layout-preserved text
- Prompt engineering with examples
- Post-processing using bounding boxes

## Key Learnings

- Prompt quality matters more than model size
- Layout context improves accuracy significantly

## Conclusion

Hybrid systems (LLM + rules) work best in production.
