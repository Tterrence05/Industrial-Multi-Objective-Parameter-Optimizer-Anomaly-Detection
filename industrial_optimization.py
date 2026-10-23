"""
Industrial Equipment Anomaly Detection and Multi-Objective Energy Optimization System
Integrates Scikit-Learn (Isolation Forest), PyMoo (NSGA-II Multi-Objective Optimization),
and Streamlit with Plotly interactive visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from datetime import datetime

# ==========================================
# 1. Page Configuration & Theme
# ==========================================
st.set_page_config(
    page_title="Industrial Digital Twin Optimizer",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2e364f;
    }
    .log-card {
        background-color: #161922;
        border-radius: 6px;
        padding: 10px;
        font-family: monospace;
        font-size: 12px;
        border-left: 4px solid #f44336;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Session State for Alert Logs
if 'alarm_logs' not in st.session_state:
    st.session_state.alarm_logs = []

def add_log(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.alarm_logs.insert(0, {"time": timestamp, "msg": message, "level": level})

# ==========================================
# 2. Data Synthesis & Model Initialization
# ==========================================
@st.cache_resource
def train_anomaly_detector(seed=42):
    np.random.seed(seed)
    n_samples = 1000
    rpm = np.random.uniform(2000, 4200, n_samples)
    flow = np.random.uniform(10, 30, n_samples)
    temp = 30 + 0.01 * rpm - 0.5 * flow + np.random.normal(0, 2, n_samples)
    vib = 5 + 0.002 * rpm + np.random.normal(0, 0.5, n_samples)
    
    X = np.column_stack([rpm, flow, temp, vib])
    model = IsolationForest(contamination=0.05, random_state=seed)
    model.fit(X)
    return model

if 'model_seed' not in st.session_state:
    st.session_state.model_seed = 42

iso_forest = train_anomaly_detector(st.session_state.model_seed)

# ==========================================
# 3. PyMoo Multi-Objective Problem Definition
# ==========================================
class IndustrialOptimizationProblem(ElementwiseProblem):
    def __init__(self, target_output, energy_cost_mult, temp_offset, model):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=1,
            xl=np.array([2000.0, 10.0]),
            xu=np.array([4200.0, 30.0])
        )
        self.target_output = target_output
        self.energy_cost_mult = energy_cost_mult
        self.temp_offset = temp_offset
        self.model = model

    def _evaluate(self, x, out, *args, **kwargs):
        rpm, flow = x[0], x[1]
        est_temp = (30 + 0.01 * rpm - 0.5 * flow) + self.temp_offset
        est_vib = 5 + 0.002 * (rpm / 1000.0) ** 2 + 0.1 * est_temp
        
        base_power = (0.005 * rpm + 0.2 * flow)
        adjusted_power = base_power * self.energy_cost_mult
        
        feature_vector = np.array([[rpm, flow, est_temp, est_vib]])
        safety_score = self.model.decision_function(feature_vector)[0]
        
        penalty = 0 if safety_score > -0.1 else abs(safety_score) * 100
        
        f1 = adjusted_power + penalty
        f2 = est_vib + penalty
        
        est_production = 0.05 * rpm
        g1 = self.target_output - est_production
        
        out["F"] = [f1, f2]
        out["G"] = [g1]

# ==========================================
# 4. Sidebar Controls & Alarm Console
# ==========================================
st.sidebar.title("Control Panel")
st.sidebar.caption("Digital Twin Configuration & Simulation")

st.sidebar.subheader("Production Target")
target_output = st.sidebar.slider("Target Output (Units)", 100, 200, 180)

st.sidebar.subheader("Optimization Hyperparameters")
pop_size = st.sidebar.slider("Population Size", 20, 150, 90)
generations = st.sidebar.slider("Generations", 10, 150, 80)

st.sidebar.markdown("---")
st.sidebar.subheader("What-If Scenario Simulator")
peak_price = st.sidebar.checkbox("Peak Electricity Price (1.5x Penalty)")
temp_offset = st.sidebar.slider("Ambient Temp Offset (°C)", -5.0, 15.0, 0.0)
energy_mult = 1.5 if peak_price else 1.0

st.sidebar.markdown("---")
st.sidebar.subheader("MLOps Model Governance")
if st.sidebar.button("Trigger Sensor Drift & Retrain Guardrail"):
    st.session_state.model_seed += 1
    train_anomaly_detector.clear()
    add_log("Sensor drift detected. Isolation Forest model retrained.", level="WARNING")
    st.sidebar.success("Guardrail Retrained Successfully!")

run_opt = st.sidebar.button("Run Multi-Objective Optimization", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("Real-time Alarm Log Console")
if st.sidebar.button("Clear Logs"):
    st.session_state.alarm_logs = []

log_container = st.sidebar.container()
with log_container:
    if not st.session_state.alarm_logs:
        st.caption("No active safety alerts. System operating normally.")
    else:
        for log in st.session_state.alarm_logs[:5]:
            color = "#f44336" if log["level"] == "ALERT" else "#ff9800" if log["level"] == "WARNING" else "#2196f3"
            st.markdown(f"""
            <div class="log-card" style="border-left-color: {color};">
                <span style="color: #888;">[{log['time']}]</span> <b>{log['level']}</b>: {log['msg']}
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 5. Header & Real-time Live Telemetry Bar
# ==========================================
st.title("Industrial Multi-Objective Parameter Optimizer")
st.caption("AI-Driven Equipment Anomaly Detection (Isolation Forest) & Evolutionary NSGA-II Multi-Objective Tuning")

with st.expander("Live Equipment Telemetry Stream (Digital Twin Monitoring)", expanded=True):
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    live_rpm = 3200 + np.random.uniform(-50, 50)
    live_flow = 16.5 + np.random.uniform(-0.5, 0.5)
    live_temp = 52.1 + np.random.uniform(-1.0, 1.0) + temp_offset
    live_vib = 11.2 + np.random.uniform(-0.2, 0.2)
    
    t_col1.metric("Live Motor RPM", f"{live_rpm:.1f} RPM")
    t_col2.metric("Cooling Flow", f"{live_flow:.1f} L/min")
    t_col3.metric("Core Temperature", f"{live_temp:.1f} °C")
    t_col4.metric("Vibration Severity", f"{live_vib:.2f} mm/s")
    
    if live_temp > 60.0:
        add_log(f"High core temperature detected: {live_temp:.1f}°C", level="ALERT")
    if live_vib > 11.3:
        add_log(f"Elevated mechanical vibration: {live_vib:.2f} mm/s", level="WARNING")

# 初始引导页
if not run_opt and 'opt_results' not in st.session_state:
    st.markdown("---")
    st.info("👈 **System Ready:** Adjust your parameters in the Control Panel on the left and click **'Run Multi-Objective Optimization'** to start.")
    
    g_col1, g_col2, g_col3 = st.columns(3)
    with g_col1:
        st.markdown("""
        <div style="background-color: #1e222d; padding: 20px; border-radius: 8px; border: 1px solid #2e364f;">
            <h4 style="color: #4CAF50; margin-top:0;">1. Telemetry & Guardrail</h4>
            <p style="font-size: 13px; color: #a0aab8;">Real-time telemetry feeds into an <b>Isolation Forest</b> safety guardrail, dynamically flagging anomalous parameter combinations.</p>
        </div>
        """, unsafe_allow_html=True)
    with g_col2:
        st.markdown("""
        <div style="background-color: #1e222d; padding: 20px; border-radius: 8px; border: 1px solid #2e364f;">
            <h4 style="color: #2196F3; margin-top:0;">2. Evolutionary Search</h4>
            <p style="font-size: 13px; color: #a0aab8;"><b>NSGA-II Algorithm</b> explores multi-objective trade-offs between minimum power usage and mechanical wear.</p>
        </div>
        """, unsafe_allow_html=True)
    with g_col3:
        st.markdown("""
        <div style="background-color: #1e222d; padding: 20px; border-radius: 8px; border: 1px solid #2e364f;">
            <h4 style="color: #FF9800; margin-top:0;">3. Pareto Export</h4>
            <p style="font-size: 13px; color: #a0aab8;">Analyze interactive 2D/3D Pareto fronts and export deduplicated optimal operational parameters for PLC/SCADA systems.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 6. Optimization Execution Engine
# ==========================================
if run_opt or 'opt_results' in st.session_state:
    if run_opt:
        add_log("Initiating NSGA-II Multi-Objective Optimization run...", level="INFO")
        problem = IndustrialOptimizationProblem(
            target_output=target_output,
            energy_cost_mult=energy_mult,
            temp_offset=temp_offset,
            model=iso_forest
        )
        
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        res = minimize(
            problem,
            algorithm,
            ('n_gen', generations),
            seed=1,
            save_history=True,
            verbose=False
        )
        st.session_state.opt_results = res
        add_log("Optimization completed successfully. Pareto set generated.", level="INFO")
    else:
        res = st.session_state.opt_results

    if res.X is not None and len(res.X) > 0:
        st.success("Optimization Completed Successfully!")
        
        X_res = res.X
        F_res = res.F
        
        rpm_res = X_res[:, 0]
        flow_res = X_res[:, 1]
        power_res = F_res[:, 0]
        vib_res = F_res[:, 1]
        temp_res = (30 + 0.01 * rpm_res - 0.5 * flow_res) + temp_offset
        
        features = np.column_stack([rpm_res, flow_res, temp_res, vib_res])
        raw_safety_scores = iso_forest.decision_function(features)
        
        # 1. Map Raw Isolation Forest Decision Scores to a 0% - 100% Health Index
        health_index = np.clip((raw_safety_scores - (-0.15)) / (0.10 - (-0.15)) * 100, 0, 100)
        
        feasible_df = pd.DataFrame({
            "RPM": rpm_res,
            "Cooling_Flow": flow_res,
            "Power_kW": power_res,
            "Vibration_mms": vib_res,
            "Temp": temp_res,
            "Safety_Score": raw_safety_scores,
            "Health_Index_%": health_index
        })
        
        baseline_rpm = 3200.0
        baseline_flow = 18.0
        baseline_power = (0.005 * baseline_rpm + 0.2 * baseline_flow) * energy_mult
        baseline_vibration = 11.2
        
        best_opt_idx = feasible_df["Power_kW"].idxmin()
        best_opt = feasible_df.iloc[best_opt_idx]
        
        # KPI Summary Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Optimal RPM", f"{best_opt['RPM']:.1f}", f"{best_opt['RPM'] - baseline_rpm:+.1f} RPM")
        kpi2.metric("Cooling Flow", f"{best_opt['Cooling_Flow']:.1f} L/min", f"{best_opt['Cooling_Flow'] - baseline_flow:+.1f}")
        kpi3.metric("Min Power Usage", f"{best_opt['Power_kW']:.2f} kW", f"{((best_opt['Power_kW'] - baseline_power)/baseline_power)*100:+.1f}%")
        kpi4.metric("Asset Health Index", f"{best_opt['Health_Index_%']:.1f}%", f"{best_opt['Health_Index_%'] - 80.0:+.1f}%")
        
        st.markdown("---")
        
        # ==========================================
        # 7. Visualizations (2D, 3D & Convergence)
        # ==========================================
        vis_tab1, vis_tab2 = st.tabs(["2D Pareto Front & Convergence", "3D Interactive Pareto Surface"])
        
        with vis_tab1:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Pareto Decision Front (Power vs Vibration)")
                st.caption("Trade-off Curve: Power Consumption vs Mechanical Wear")
                
                fig_pareto = px.scatter(
                    feasible_df,
                    x="Power_kW",
                    y="Vibration_mms",
                    color="Health_Index_%",
                    size="RPM",
                    hover_data=["RPM", "Cooling_Flow", "Temp"],
                    labels={"Power_kW": "Power Consumption (kW)", "Vibration_mms": "Vibration (mm/s)"}
                )
                fig_pareto.add_trace(go.Scatter(
                    x=[baseline_power], y=[baseline_vibration],
                    mode="markers+text",
                    marker=dict(size=14, color="red", symbol="x"),
                    text=["Baseline"], textposition="top center",
                    name="Unoptimized Baseline"
                ))
                fig_pareto.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_pareto, use_container_width=True)
                
            with col_right:
                st.subheader("Evolutionary Convergence Curve")
                history = res.history
                gens = np.arange(len(history))
                min_power = [np.min(h.opt.get("F")[:, 0]) for h in history]

                fig_conv = go.Figure()
                fig_conv.add_trace(go.Scatter(x=gens, y=min_power, mode='lines+markers', name='Best Power Score'))
                fig_conv.update_layout(
                    title="NSGA-II Convergence History",
                    xaxis_title="Generation",
                    yaxis_title="Minimum Objective Value (kW)",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_conv, use_container_width=True)

        with vis_tab2:
            st.subheader("High-Dimensional 3D Trade-off Surface")
            st.caption("Rotate and zoom to explore Power (X), Vibration (Y), and Motor RPM (Z)")
            
            fig_3d = px.scatter_3d(
                feasible_df,
                x="Power_kW",
                y="Vibration_mms",
                z="RPM",
                color="Health_Index_%",
                size="Cooling_Flow",
                hover_data=["Temp"],
                labels={"Power_kW": "Power (kW)", "Vibration_mms": "Vibration (mm/s)", "RPM": "Motor RPM"}
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_3d, use_container_width=True)

        # ==========================================
        # 8. Solution Options Table & ROI Analysis
        # ==========================================
        st.subheader("Pareto Solution Trade-off Options")
        
        display_df = feasible_df[["RPM", "Cooling_Flow", "Power_kW", "Vibration_mms", "Health_Index_%"]].copy()
        display_df["RPM"] = display_df["RPM"].round(1)
        display_df["Cooling_Flow"] = display_df["Cooling_Flow"].round(1)
        display_df["Power_kW"] = display_df["Power_kW"].round(2)
        display_df["Vibration_mms"] = display_df["Vibration_mms"].round(2)
        display_df["Health_Index_%"] = display_df["Health_Index_%"].round(1)
        
        display_df = (
            display_df.drop_duplicates(subset=["RPM", "Cooling_Flow", "Power_kW", "Vibration_mms"])
            .sort_values(by="Power_kW")
            .reset_index(drop=True)
        )
        
        selected_mode = st.radio(
            "Select Operational Strategy Preset:",
            options=["Min Power Mode (Energy Saving)", "Min Vibration Mode (Equipment Protection)", "Balanced Trade-off Mode"],
            horizontal=True
        )
        
        if "Energy Saving" in selected_mode:
            preset_idx = display_df["Power_kW"].idxmin()
        elif "Equipment Protection" in selected_mode:
            preset_idx = display_df["Vibration_mms"].idxmin()
        else:
            preset_idx = len(display_df) // 2
            
        selected_row = display_df.iloc[preset_idx]
        st.info(f"**Current Strategy Applied:** RPM = **{selected_row['RPM']}**, Cooling Flow = **{selected_row['Cooling_Flow']} L/min** (Power: **{selected_row['Power_kW']} kW**, Vibration: **{selected_row['Vibration_mms']} mm/s**)")

        st.dataframe(display_df, use_container_width=True)

        # 🕸️ 交互式多策略雷达图 (Strategy Trade-off Radar Chart)
        st.markdown("---")
        st.subheader("Multi-Strategy Trade-off Radar Comparison")
        st.caption("Visualizing operational focus across energy efficiency, equipment preservation, and asset safety.")

        # 提取三种预设模式的数据点
        min_p_row = display_df.iloc[display_df["Power_kW"].idxmin()]
        min_v_row = display_df.iloc[display_df["Vibration_mms"].idxmin()]
        bal_row = display_df.iloc[len(display_df) // 2]

        categories = ['Energy Efficiency', 'Vibration Suppression', 'Asset Health Index', 'Cooling Overhead']

        def calc_radar_scores(row):
            p_score = np.clip((30.0 - row["Power_kW"]) / (30.0 - 20.0) * 100, 10, 100)
            v_score = np.clip((13.0 - row["Vibration_mms"]) / (13.0 - 10.0) * 100, 10, 100)
            h_score = row["Health_Index_%"]
            c_score = np.clip((30.0 - row["Cooling_Flow"]) / (30.0 - 10.0) * 100, 10, 100)
            return [p_score, v_score, h_score, c_score]

        fig_radar = go.Figure()

        fig_radar.add_trace(go.Scatterpolar(
            r=calc_radar_scores(min_p_row),
            theta=categories,
            fill='toself',
            name='Min Power Mode'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=calc_radar_scores(min_v_row),
            theta=categories,
            fill='toself',
            name='Min Vibration Mode'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=calc_radar_scores(bal_row),
            theta=categories,
            fill='toself',
            name='Balanced Trade-off Mode'
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            margin=dict(l=40, r=40, t=20, b=20)
        )

        st.plotly_chart(fig_radar, use_container_width=True)

        # 💰 商业 ROI 年化收益预测分析
        st.markdown("---")
        st.subheader("Estimated Business ROI & Impact Analysis")
        st.caption("Projected annual financial savings and mechanical lifespan extension compared to baseline execution.")
        
        electricity_rate = 0.15 # $0.15 per kWh
        operating_hours = 8760  # 24/7 annual operation
        
        power_saved_kw = baseline_power - selected_row["Power_kW"]
        annual_financial_impact = power_saved_kw * operating_hours * electricity_rate
        
        vib_reduction_pct = ((baseline_vibration - selected_row["Vibration_mms"]) / baseline_vibration) * 100
        est_lifespan_increase = vib_reduction_pct * 1.2
        
        roi_col1, roi_col2, roi_col3 = st.columns(3)
        
        if annual_financial_impact >= 0:
            roi_col1.metric("Projected Annual Energy Savings", f"${annual_financial_impact:,.2f} / yr", f"{power_saved_kw:+.2f} kW / hr")
        else:
            roi_col1.metric("Projected Annual Extra Cost", f"${abs(annual_financial_impact):,.2f} / yr", f"{power_saved_kw:+.2f} kW / hr")
            
        roi_col2.metric("Vibration Reduction Rate", f"{vib_reduction_pct:+.1f}%", "Reduced Wear & Tear" if vib_reduction_pct >= 0 else "Increased Wear")
        roi_col3.metric("Equipment Lifespan Impact", f"{est_lifespan_increase:+.1f}%", "Lower Maintenance Cost" if est_lifespan_increase >= 0 else "Higher Maintenance Cost")

        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Optimized Parameters (CSV)",
            data=csv_data,
            file_name="optimized_pareto_parameters.csv",
            mime="text/csv"
        )

        # 🔌 PLC / SCADA 控制命令 Payload
        with st.expander("🔌 Export PLC/SCADA Control Command (Modbus / MQTT Payload)"):
            st.caption("Standardized control payload ready for OT network transmission via MQTT or OPC-UA.")
            control_payload = {
                "timestamp": datetime.now().isoformat(),
                "target_equipment_id": "PUMP_ASSET_01",
                "applied_mode": selected_mode.split("(")[0].strip(),
                "control_parameters": {
                    "setpoint_rpm": float(selected_row["RPM"]),
                    "cooling_water_flow_lpm": float(selected_row["Cooling_Flow"])
                },
                "safety_guardrail": {
                    "health_index_pct": float(selected_row["Health_Index_%"]),
                    "isolation_forest_status": "PASSED" if selected_row["Health_Index_%"] > 40 else "WARNING"
                }
            }
            st.json(control_payload)

        # 📄 生成诊断与优化总结报告
        st.markdown("---")
        st.subheader("System Diagnostic Summary Report")
        
        report_text = f"""
=====================================================
INDUSTRIAL DIGITAL TWIN OPTIMIZATION REPORT
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=====================================================

1. OPERATIONAL STRATEGY
-----------------------------------------------------
- Selected Strategy: {selected_mode}
- Target Production Output: {target_output} Units
- Motor RPM Setpoint: {selected_row['RPM']} RPM
- Cooling Flow Setpoint: {selected_row['Cooling_Flow']} L/min

2. PERFORMANCE METRICS
-----------------------------------------------------
- Estimated Power Consumption: {selected_row['Power_kW']} kW
- Mechanical Vibration Level: {selected_row['Vibration_mms']} mm/s
- Asset Health Index: {selected_row['Health_Index_%']}%

3. PROJECTED BUSINESS ROI IMPACT
-----------------------------------------------------
- Annual Financial Impact: ${annual_financial_impact:,.2f} / yr
- Mechanical Lifespan Impact: {est_lifespan_increase:+.1f}%
=====================================================
        """
        
        st.download_button(
            label="📄 Download Executive Summary Report (.txt)",
            data=report_text,
            file_name=f"digital_twin_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    else:
        st.error("Optimization failed to find valid parameters.")