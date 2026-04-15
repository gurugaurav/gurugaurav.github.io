---
layout: post
title: "Real-Time Chart Pattern Detection using YOLO"
date: 2026-04-17
categories: cv trading
---

## Idea

Detect chart patterns like head & shoulders from live screen capture.

## System Design

- Capture screen every 5 seconds
- Run YOLO model
- Display bounding boxes live

## Challenges

- Labeling training data
- Handling noisy charts

## Result

Achieved near real-time detection with visual feedback.
