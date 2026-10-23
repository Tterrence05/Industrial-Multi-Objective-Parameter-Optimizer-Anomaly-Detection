
# ⚙️ Industrial Digital Twin Parameter Optimizer & Decision Support System

An enterprise-grade, high-fidelity Industrial Digital Twin dashboard engineered for real-time equipment monitoring, predictive maintenance guardrails, and multi-objective parameter optimization.

This platform integrates **Evolutionary Multi-Objective Optimization (NSGA-II)**, **Machine Learning Anomaly Detection (Isolation Forest)**, **What-If Stress Testing**, and **OT/IT Integration Payloads (Modbus/MQTT JSON)** into a streamlined Streamlit Decision Support System (DSS).

---

## 📐 System Architecture

```mermaid
graph TD
    A[Real-Time Telemetry Stream\nRPM, Flow, Temp, Vib] --> B[Isolation Forest Guardrail\nAnomaly Detection & Health Score]
    B --> C{Safety Constraint Check}
    C -->|Valid / Penalty Factor| D[NSGA-II Multi-Objective Solver\nPyMoo Engine]
    E[User Inputs & What-If Scenarios\nTarget Output, Electricity Cost, Ambient Temp] --> D
    D --> F[Pareto Optimal Solution Set\n2D/3D Pareto Surfaces]
    F --> G[MCDM & Strategy Selector\nMin Power, Min Vibration, Balanced]
    G --> H[Financial ROI & Lifespan Estimator]
    G --> I[PLC/SCADA Control Payload Export\nMQTT / Modbus JSON]
    G --> J[Executive Summary Report Generator]

✨ Key Features
Multi-Objective Pareto Solver: Driven by NSGA-II (pymoo), balancing power consumption (kW) against mechanical vibration severity ($mm/s$) subject to production target constraints.
MLOps Model Governance & Guardrail: Uses Isolation Forest anomaly detection to calculate a normalized Asset Health Index (0–100%) and includes a real-time trigger for sensor drift simulation and dynamic model retraining.
Interactive 2D/3D & Radar Analytics: Visualizes multi-dimensional trade-offs using Plotly interactive scatter plots, 3D Pareto surfaces, and multi-strategy radar charts.
Financial ROI & Mechanical Impact Analysis: Estimates annual energy cost savings ($/yr) and projected equipment lifespan extensions (+%) based on selected operational strategies.
OT/IT Integration Payload: Generates standardized JSON control payloads tailored for direct transmission over industrial protocols (Modbus / MQTT / OPC-UA).

🛠️ Tech Stack & Dependencies

Frontend & Dashboard: Streamlit

Optimization & Decision Science: PyMoo (NSGA-II), Scikit-Learn (Isolation Forest), NumPy, Pandas

Interactive Visualization: Plotly Express, Plotly Graph Objects

System Architecture Visuals: Mermaid.js
