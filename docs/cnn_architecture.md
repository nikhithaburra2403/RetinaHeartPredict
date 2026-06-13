# Retina CNN Architecture

The CNN model in app/ml/retina_cnn.py is designed for retinal image analysis with an input shape of 224 x 224 x 3.

Architecture summary:
1. Input layer: accepts RGB retinal images at 224x224.
2. Conv2D blocks: learn low-level edges, texture, and vessel-like patterns.
3. MaxPooling layers: reduce spatial size and keep the strongest activations.
4. BatchNormalization: stabilizes and accelerates training.
5. Dropout layers: reduce overfitting.
6. Flatten layer: converts feature maps into a vector.
7. Dense feature layer: produces a compact representation for downstream classification.
8. Softmax output: predicts one of two classes (for example, healthy vs at risk).

This structure is suitable as a feature extractor or as a full classifier for retinal image analysis.
