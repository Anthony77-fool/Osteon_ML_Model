# Osteon ML Model: Relative Change-Based Adaptive Mastery Engine

An intelligent, dynamic adaptive learning engine designed to predict relative student mastery changes ($\Delta$) in a VR Anatomy application environment. By transitioning from absolute mastery predictions to a change-based relative model ($\Delta$), this system eliminates performance prediction plateaus and dynamically adjusts scores based on difficulty and response speed.


# 🛠️ Tech Stack & Dependencies
To execute the Python pipeline, build the ONNX model, and run inferences, configure your environment with the following dependencies.

Prerequisites & Console Installation
Ensure you have Python 3.9+ installed, then install the required packages via pip:

```bash
pip install pandas numpy scikit-learn skl2onnx onnxruntime
```

| Package / Tool | Purpose |
|---|---|
| **Python 3.9+** | Execution environment |
| **Scikit-Learn** | Pipeline building, preprocessor transformations, and Random Forest regressor training |
| **skl2onnx** | Converting trained Scikit-Learn pipelines into optimized `.onnx` model files |
| **onnxruntime** | Lightweight, high-performance C++/Python inference engine used in execution loops |
| **Pandas & NumPy** | Dataset manipulation, feature engineering, and mathematical matrix operations |
| **Google Colab** | Cloud-based notebook environment used for training and exporting models |


# 💡 System Logic & Adaptive Loop Explanation

### 1. The Core Philosophy

Instead of training a Machine Learning model to guess an absolute, total mastery score (which causes stagnation and prediction plateaus), this pipeline predicts **the relative change in mastery ($\Delta$)** earned on a single question attempt.

### 2. How the Inference Loop Works

1. **Input State ($X$)**: The system accepts the user's previous mastery level ($M_{t-1}$), response time in seconds, item difficulty score, correctness ($1$ or $0$), and metadata.

2. **Model Prediction ($y$)**: The model evaluates performance features and outputs a predicted delta ($y = \Delta$), representing the exact points to gain or lose.

3. **Safety Guardrails**: Logic rules ensure correct answers yield positive deltas ($\Delta > 0$) while incorrect answers yield negative deltas ($\Delta < 0$).

4. **State Update**: The previous mastery score ($M_{t-1}$) is updated with the predicted delta ($y$) to generate the new mastery score ($M_t$), which is then clamped to remain within $[0.0, 1.0]$.

5. **State Propagation**: $M_t$ is saved and passed forward as $M_{t-1}$ for the next attempt step ($t+1$).

# 📐 Mathematical Model & Formulas

The adaptive loop governs state updates using a sequential recursive function bounded between $0.0$ and $1.0$.

### 1. State Update Equation

The core mastery update formula for step $t$ is expressed as:

<div align="center">

\[
M_t = \operatorname{clip}(M_{t-1} + y,\; 0.0,\; 1.0)
\]

</div>
