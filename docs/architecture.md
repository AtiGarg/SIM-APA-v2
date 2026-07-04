
# SIM-APA v2.0 Architecture

## Goal

Build a clean, modular, reproducible research codebase for:

Semantic Importance-Guided DeepJSCC with Adaptive Power Allocation
for Object-Centric Vehicular Semantic Communications.

## Core Pipeline

Input Image
-> YOLOv10 Backbone
-> P5 Feature Map (256 x 20 x 20)
-> Semantic Importance Module
-> Adaptive Power Allocation
-> DeepJSCC Encoder
-> Wireless Channel
-> DeepJSCC Decoder
-> Recovered Feature Map
-> Evaluation

## Milestones

1. Foundation
2. Semantic Feature Extraction
3. Semantic Importance Module
4. Adaptive Power Allocation
5. DeepJSCC and Wireless Channels
6. Training
7. Evaluation and Publication Figures

## Development Rules

1. No notebook spaghetti code.
2. All real implementation goes inside Python modules.
3. Every module must be documented.
4. Every milestone must be tested before moving forward.
5. Paper methodology is the source of truth.
6. Configuration values must not be hardcoded.
7. Code must be GitHub-release ready.
