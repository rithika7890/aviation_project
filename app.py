from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

app = Flask(__name__)

# load trained model
model = joblib.load("android app/models/final_rul_model.pkl")

# SHAP explainer
explainer = shap.TreeExplainer(model)

fleet_data = None


# -----------------------------
# Fleet Monitoring Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    global fleet_data
    fleet_table = None

    if request.method == "POST":

        file = request.files["file"]

        df = pd.read_csv(file, header=None)

        columns = ["engine_id", "cycle"]
        columns += [f"op_setting_{i}" for i in range(1,4)]
        columns += [f"sensor_{i}" for i in range(1,22)]

        df.columns = columns

        fleet_data = df

        engines = sorted(df["engine_id"].unique())

        results = []

        for engine in engines:

            df_engine = df[df["engine_id"] == engine].copy()

            # rolling mean features
            for i in range(1,22):
                df_engine[f"sensor_{i}_mean"] = df_engine[f"sensor_{i}"].rolling(5).mean()

            df_engine = df_engine.fillna(method="bfill")

            X = df_engine.tail(1).drop(columns=["engine_id","cycle"])

            rul = int(model.predict(X)[0])

            # health status
            if rul > 80:
                status = "Healthy"
            elif rul > 30:
                status = "Warning"
            else:
                status = "Critical"

            results.append({
                "engine": int(engine),
                "rul": rul,
                "status": status
            })

        fleet_table = results

    return render_template("index.html", fleet=fleet_table)



# -----------------------------
# Engine Dashboard
# -----------------------------
@app.route("/engine/<int:engine_id>")
def engine_dashboard(engine_id):

    global fleet_data

    df_engine = fleet_data[fleet_data["engine_id"] == engine_id].copy()

    for i in range(1,22):
        df_engine[f"sensor_{i}_mean"] = df_engine[f"sensor_{i}"].rolling(5).mean()

    df_engine = df_engine.fillna(method="bfill")

    X = df_engine.tail(1).drop(columns=["engine_id","cycle"])

    prediction = int(model.predict(X)[0])

    current_cycle = int(df_engine["cycle"].max())

    failure_cycle = current_cycle + prediction


    # -----------------------------
    # RUL Gauge
    # -----------------------------
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Remaining Useful Life"},
        gauge={
            'axis': {'range': [0,200]},
            'steps': [
                {'range':[0,30],'color':"red"},
                {'range':[30,80],'color':"orange"},
                {'range':[80,200],'color':"green"}
            ]
        }
    ))

    gauge = gauge_fig.to_html(full_html=False)



    # -----------------------------
    # Improved Degradation Curve
    # -----------------------------

    history_cycles = df_engine["cycle"].values

    history_rul = prediction * np.exp(-history_cycles / (current_cycle + 1))

    future_cycles = np.linspace(current_cycle, failure_cycle, 50)

    future_rul = prediction * np.exp(-(future_cycles - current_cycle) / prediction)

    future_rul = np.clip(future_rul, 0, None)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history_cycles,
        y=history_rul,
        mode='lines',
        name="Observed Degradation",
        line=dict(color="blue", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=future_cycles,
        y=future_rul,
        mode='lines',
        name="Predicted Failure",
        line=dict(color="black", dash="dash", width=3)
    ))

    # Healthy zone
    fig.add_hrect(
        y0=80,
        y1=200,
        fillcolor="green",
        opacity=0.15,
        layer="below",
        line_width=0,
    )

    # Warning zone
    fig.add_hrect(
        y0=30,
        y1=80,
        fillcolor="orange",
        opacity=0.15,
        layer="below",
        line_width=0,
    )

    # Critical zone
    fig.add_hrect(
        y0=0,
        y1=30,
        fillcolor="red",
        opacity=0.15,
        layer="below",
        line_width=0,
    )

    fig.update_layout(
        title="Engine Degradation Curve with Health Zones",
        xaxis_title="Engine Cycle",
        yaxis_title="Remaining Useful Life",
        height=500
    )

    curve = fig.to_html(full_html=False)



    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "value": shap_values[0]
    })

    shap_df["importance"] = shap_df["value"].abs()

    shap_df = shap_df.sort_values(
        "importance",
        ascending=False
    ).head(10)

    shap_fig = px.bar(
        shap_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top Features Influencing Prediction"
    )

    shap_chart = shap_fig.to_html(full_html=False)



    return render_template(
        "engine_dashboard.html",
        engine=engine_id,
        prediction=prediction,
        current_cycle=current_cycle,
        failure_cycle=failure_cycle,
        gauge=gauge,
        curve=curve,
        shap=shap_chart
    )


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)