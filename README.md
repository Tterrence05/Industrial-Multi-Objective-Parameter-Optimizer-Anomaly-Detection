# ⚙️ Industrial Digital Twin Parameter Optimizer & Decision Support System

An enterprise-grade, high-fidelity Industrial Digital Twin dashboard engineered for real-time equipment monitoring, predictive maintenance guardrails, and multi-objective parameter optimization.

This platform integrates **Evolutionary Multi-Objective Optimization (NSGA-II)**, **Machine Learning Anomaly Detection (Isolation Forest)**, **What-If Stress Testing**, and **OT/IT Integration Payloads (Modbus/MQTT JSON)** into a streamlined Streamlit Decision Support System (DSS).

---

## 📐 System Architecture

```mermaid
flowchart TD
    A["📡 Real-Time Telemetry Stream<br>(RPM, Flow, Temp, Vib)"] --> B["🛡️ Isolation Forest Guardrail<br>(Anomaly Detection & Health Score)"]
    B --> C{"⚠️ Safety Constraint Check"}
    C -->|"Valid / Penalty Factor"| D["🧬 NSGA-II Multi-Objective Solver<br>(PyMoo Engine)"]
    E["🎛️ User Inputs & What-If Scenarios<br>(Target Output, Price, Temp)"] --> D
    D --> F["📈 Pareto Optimal Solution Set<br>(2D/3D Pareto Surfaces)"]
    F --> G["🎯 MCDM & Strategy Selector<br>(Min Power, Min Vibration, Balanced)"]
    G --> H["💰 Financial ROI & Lifespan Estimator"]
    G --> I["🔌 PLC/SCADA Control Payload Export<br>(MQTT / Modbus JSON)"]
    G --> J["📄 Executive Summary Report Generator"]
