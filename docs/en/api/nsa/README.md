---
id: nsa
sidebar_label: aisp.nsa
keywords: 
    - immune
    - artificial immune systems
    - classification
    - negative selection
    - binary features
    - anomaly detection
    - non-self recognition
    - pattern recognition
    - multiclass
    - real-valued
    - v-detector
---

# aisp.nsa

Module (NSA) Negative Selection Algorithm.

> **Module:** `aisp.nsa`

## Overview

NSAs simulate the maturation process of T-cells in the immune system, where these cells learn to
distinguish between self and non-self. Only T-cells capable of recognizing non-self elements are
preserved.

## Classes

| Class               | Description                                                                         |
|---------------------|-------------------------------------------------------------------------------------|
| [`RNSA`](./rnsa.md) | A supervised learning algorithm for classification that uses real-valued detectors. |
| [`BNSA`](./bnsa.md) | A supervised learning algorithm for classification that uses binary detectors.      |
