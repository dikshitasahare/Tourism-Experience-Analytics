"""
app.py - Flask application for Tourism Experience Analytics
Run: python app.py
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
import os

app = Flask(__name__)

# ---------------------------------------------------------------
# Load data and trained models once at startup
# ---------------------------------------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

transactions_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))
attractions_df = pd.read_csv(os.path.join(DATA_DIR, "attractions.csv"))

le_country = joblib.load(os.path.join(MODEL_DIR, "le_country.pkl"))
le_city = joblib.load(os.path.join(MODEL_DIR, "le_city.pkl"))
le_attraction = joblib.load(os.path.join(MODEL_DIR, "le_attraction.pkl"))
le_visitmode = joblib.load(os.path.join(MODEL_DIR, "le_visitmode.pkl"))

rating_model = joblib.load(os.path.join(MODEL_DIR, "rating_model.pkl"))
visitmode_model = joblib.load(os.path.join(MODEL_DIR, "visitmode_model.pkl"))
item_sim_df = joblib.load(os.path.join(MODEL_DIR, "item_similarity.pkl"))

# Dropdown option lists, built fresh from the data
COUNTRIES = sorted(transactions_df["Country"].unique())
CITY_MAP = transactions_df.groupby("Country")["City"].unique().apply(list).to_dict()
ATTRACTION_MAP = transactions_df.groupby("City")["Attraction"].unique().apply(list).to_dict()
ALL_ATTRACTIONS = sorted(transactions_df["Attraction"].unique())
MONTHS = list(range(1, 13))

# ---------------------------------------------------------------
# Lat/Lon lookup for Indian cities (extend this with any city
# names that appear in your dataset but aren't listed here)
# ---------------------------------------------------------------
INDIA_CITY_COORDS = {
    "Delhi": (28.7041, 77.1025),
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    "Goa": (15.2993, 74.1240),
    "Agra": (27.1767, 78.0081),
    "Varanasi": (25.3176, 82.9739),
    "Surat": (21.1702, 72.8311),
    "Kochi": (9.9312, 76.2673),
    "Cochin": (9.9312, 76.2673),
    "Chandigarh": (30.7333, 76.7794),
    "Indore": (22.7196, 75.8577),
    "Nagpur": (21.1458, 79.0882),
    "Bhopal": (23.2599, 77.4126),
    "Patna": (25.5941, 85.1376),
    "Amritsar": (31.6340, 74.8723),
    "Udaipur": (24.5854, 73.7125),
    "Shimla": (31.1048, 77.1734),
    "Manali": (32.2432, 77.1892),
    "Rishikesh": (30.0869, 78.2676),
    "Darjeeling": (27.0410, 88.2663),
    "Mysore": (12.2958, 76.6394),
    "Mysuru": (12.2958, 76.6394),
    "Coimbatore": (11.0168, 76.9558),
    "Visakhapatnam": (17.6868, 83.2185),
    "Nashik": (19.9975, 73.7898),
    "Vadodara": (22.3072, 73.1812),
    "Ranchi": (23.3441, 85.3096),
    "Bhubaneswar": (20.2961, 85.8245),
    "Guwahati": (26.1445, 91.7362),
    "Dehradun": (30.3165, 78.0322),
    "Jodhpur": (26.2389, 73.0243),
    "Pondicherry": (11.9416, 79.8083),
    "Puducherry": (11.9416, 79.8083),
}


def stars_html(rating):
    """Convert numeric rating (0-5) into a filled/empty star string."""
    full = int(round(rating))
    full = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full)


# ---------------------------------------------------------------
# HOME
# ---------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------------------
# RATING PREDICTION
# ---------------------------------------------------------------
@app.route("/rating", methods=["GET", "POST"])
def rating_prediction():
    result = None
    selected = {}

    if request.method == "POST":
        country = request.form.get("country")
        city = request.form.get("city")
        attraction = request.form.get("attraction")
        month = int(request.form.get("month"))

        selected = {"country": country, "city": city, "attraction": attraction, "month": month}

        try:
            country_enc = le_country.transform([country])[0]
            city_enc = le_city.transform([city])[0]
            attraction_enc = le_attraction.transform([attraction])[0]

            features = [[country_enc, city_enc, attraction_enc, month]]
            predicted_rating = rating_model.predict(features)[0]
            predicted_rating = round(float(predicted_rating), 1)

            result = {
                "rating": predicted_rating,
                "stars": stars_html(predicted_rating)
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "rating.html",
        countries=COUNTRIES,
        city_map=CITY_MAP,
        attraction_map=ATTRACTION_MAP,
        months=MONTHS,
        result=result,
        selected=selected
    )


# ---------------------------------------------------------------
# VISIT MODE PREDICTION
# ---------------------------------------------------------------
@app.route("/visitmode", methods=["GET", "POST"])
def visitmode_prediction():
    result = None
    selected = {}

    if request.method == "POST":
        country = request.form.get("country")
        city = request.form.get("city")
        attraction = request.form.get("attraction")
        month = int(request.form.get("month"))
        rating = float(request.form.get("rating"))

        selected = {
            "country": country, "city": city, "attraction": attraction,
            "month": month, "rating": rating
        }

        try:
            country_enc = le_country.transform([country])[0]
            city_enc = le_city.transform([city])[0]
            attraction_enc = le_attraction.transform([attraction])[0]

            features = [[country_enc, city_enc, attraction_enc, month, rating]]
            pred_enc = visitmode_model.predict(features)[0]
            predicted_mode = le_visitmode.inverse_transform([pred_enc])[0]

            result = {"mode": predicted_mode}
        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "visitmode.html",
        countries=COUNTRIES,
        city_map=CITY_MAP,
        attraction_map=ATTRACTION_MAP,
        months=MONTHS,
        result=result,
        selected=selected
    )


# ---------------------------------------------------------------
# RECOMMENDATIONS (item-based collaborative filtering)
# ---------------------------------------------------------------
@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    recommendations = []
    selected_attraction = None

    if request.method == "POST":
        selected_attraction = request.form.get("attraction")
        if selected_attraction in item_sim_df.columns:
            sims = item_sim_df[selected_attraction].drop(labels=[selected_attraction])
            top5 = sims.sort_values(ascending=False).head(5)
            recommendations = list(top5.index)

    return render_template(
        "recommend.html",
        attractions=ALL_ATTRACTIONS,
        recommendations=recommendations,
        selected_attraction=selected_attraction
    )

# ---------------------------------------------------------------
# BEST TIME TO VISIT
# ---------------------------------------------------------------
@app.route("/besttime", methods=["GET", "POST"])
def best_time():
    result = None
    selected_attraction = None
    chart_html = None

    if request.method == "POST":
        selected_attraction = request.form.get("attraction")

        attr_df = transactions_df[transactions_df["Attraction"] == selected_attraction]

        if not attr_df.empty:
            monthly_avg = (
                attr_df.groupby("VisitMonth")["Rating"]
                .mean()
                .reset_index()
                .rename(columns={"Rating": "AvgRating"})
            )
            monthly_avg["AvgRating"] = monthly_avg["AvgRating"].round(2)

            best_row = monthly_avg.loc[monthly_avg["AvgRating"].idxmax()]
            month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]

            result = {
                "best_month": month_names[int(best_row["VisitMonth"]) - 1],
                "best_rating": best_row["AvgRating"],
                "total_visits": int(len(attr_df)),
            }

            fig = px.bar(
                monthly_avg, x="VisitMonth", y="AvgRating",
                title="",
                color="AvgRating",
                color_continuous_scale=["#90caf9", "#1a2b4c"],
            )
            fig.update_traces(marker_line_width=0)
            fig.update_layout(
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="Segoe UI", color="#1a2b4c", size=12),
                margin=dict(l=40, r=20, t=20, b=50),
                coloraxis_showscale=False,
                height=360,
            )
            fig.update_xaxes(
                title_text="Month", showgrid=False,
                tickmode="array",
                tickvals=list(range(1, 13)),
                ticktext=month_names,
            )
            fig.update_yaxes(
                title_text="Avg Rating", showgrid=True,
                gridcolor="rgba(26,43,76,0.08)",
            )
            chart_html = pio.to_html(
                fig, full_html=False, include_plotlyjs=False,
                config={"responsive": True, "displayModeBar": False},
            )

    return render_template(
        "best_time.html",
        attractions=ALL_ATTRACTIONS,
        selected_attraction=selected_attraction,
        result=result,
        chart_html=chart_html,
    )

# ---------------------------------------------------------------
# COMPARE ATTRACTIONS
# ---------------------------------------------------------------
@app.route("/compare", methods=["GET", "POST"])
def compare_attractions():
    result = None
    selected = {}

    if request.method == "POST":
        attr_a = request.form.get("attraction_a")
        attr_b = request.form.get("attraction_b")
        selected = {"a": attr_a, "b": attr_b}

        if attr_a and attr_b and attr_a != attr_b:
            df_a = transactions_df[transactions_df["Attraction"] == attr_a]
            df_b = transactions_df[transactions_df["Attraction"] == attr_b]

            def build_stats(df):
                if df.empty:
                    return None
                mode_counts = df["VisitMode"].value_counts()
                top_mode = mode_counts.idxmax() if not mode_counts.empty else "N/A"
                return {
                    "avg_rating": round(float(df["Rating"].mean()), 2),
                    "total_visits": int(len(df)),
                    "top_mode": top_mode,
                    "top_mode_pct": round(mode_counts.max() / len(df) * 100, 1)
                                    if not mode_counts.empty else 0,
                    "cities": int(df["City"].nunique()),
                    "countries": int(df["Country"].nunique()),
                    "stars": stars_html(df["Rating"].mean()),
                }

            stats_a = build_stats(df_a)
            stats_b = build_stats(df_b)

            if stats_a and stats_b:
                # comparison bar chart for ratings
                fig = px.bar(
                    x=[attr_a, attr_b],
                    y=[stats_a["avg_rating"], stats_b["avg_rating"]],
                    color=[attr_a, attr_b],
                    color_discrete_sequence=["#1a2b4c", "#4fc3f7"],
                    labels={"x": "Attraction", "y": "Avg Rating"},
                )
                fig.update_traces(marker_line_width=0, showlegend=False)
                fig.update_layout(
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="Segoe UI", color="#1a2b4c", size=12),
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=320,
                    showlegend=False,
                )
                fig.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                                 title_text="Avg Rating", range=[0, 5])
                fig.update_xaxes(showgrid=False, title_text="")
                chart_html = pio.to_html(
                    fig, full_html=False, include_plotlyjs=False,
                    config={"responsive": True, "displayModeBar": False},
                )

                result = {
                    "a": stats_a, "b": stats_b,
                    "winner": attr_a if stats_a["avg_rating"] >= stats_b["avg_rating"] else attr_b,
                    "chart_html": chart_html,
                }

    return render_template(
        "compare.html",
        attractions=ALL_ATTRACTIONS,
        selected=selected,
        result=result,
    )


    # ---------------------------------------------------------------
# SEARCH ATTRACTION
# ---------------------------------------------------------------
@app.route("/search", methods=["GET"])
def search_attraction():
    query = request.args.get("q", "").strip()
    result = None
    suggestions = []

    if query:
        # find exact or closest match (case-insensitive)
        matches = [a for a in ALL_ATTRACTIONS if query.lower() in a.lower()]

        if matches:
            # if exact match exists, use it; else use first partial match
            exact = [a for a in matches if a.lower() == query.lower()]
            chosen = exact[0] if exact else matches[0]
            suggestions = matches[:8]

            df = transactions_df[transactions_df["Attraction"] == chosen]

            if not df.empty:
                mode_counts = df["VisitMode"].value_counts()
                monthly = df.groupby("VisitMonth").size().reset_index(name="Visits")
                month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"]
                best_month_row = (
                    df.groupby("VisitMonth")["Rating"].mean().idxmax()
                    if not df.empty else None
                )

                fig = px.bar(
                    monthly, x="VisitMonth", y="Visits",
                    color="Visits",
                    color_continuous_scale=["#90caf9", "#1a2b4c"],
                )
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="Segoe UI", color="#1a2b4c", size=12),
                    margin=dict(l=40, r=20, t=20, b=50),
                    coloraxis_showscale=False,
                    height=300,
                )
                fig.update_xaxes(
                    title_text="Month", showgrid=False,
                    tickmode="array", tickvals=list(range(1, 13)), ticktext=month_names,
                )
                fig.update_yaxes(title_text="Visits", showgrid=True,
                                 gridcolor="rgba(26,43,76,0.08)")
                chart_html = pio.to_html(
                    fig, full_html=False, include_plotlyjs=False,
                    config={"responsive": True, "displayModeBar": False},
                )

                result = {
                    "name": chosen,
                    "avg_rating": round(float(df["Rating"].mean()), 2),
                    "stars": stars_html(df["Rating"].mean()),
                    "total_visits": int(len(df)),
                    "cities": sorted(df["City"].unique().tolist()),
                    "countries": sorted(df["Country"].unique().tolist()),
                    "top_mode": mode_counts.idxmax() if not mode_counts.empty else "N/A",
                    "top_mode_pct": round(mode_counts.max() / len(df) * 100, 1)
                                    if not mode_counts.empty else 0,
                    "best_month": month_names[int(best_month_row) - 1]
                                  if best_month_row else "N/A",
                    "chart_html": chart_html,
                }

    return render_template(
        "search.html",
        query=query,
        result=result,
        suggestions=suggestions,
        all_attractions=ALL_ATTRACTIONS,
    )


    # ---------------------------------------------------------------
# COUNTRY EXPLORER
# ---------------------------------------------------------------
@app.route("/country-explorer", methods=["GET", "POST"])
def country_explorer():
    result = None
    selected_country = None

    if request.method == "POST":
        selected_country = request.form.get("country")
        df = transactions_df[transactions_df["Country"] == selected_country]

        if not df.empty:
            top_cities = (
                df["City"].value_counts().head(8).reset_index()
            )
            top_cities.columns = ["City", "Visits"]

            top_attractions = (
                df["Attraction"].value_counts().head(6).reset_index()
            )
            top_attractions.columns = ["Attraction", "Visits"]

            mode_counts = df["VisitMode"].value_counts()

            # city bar chart
            fig1 = px.bar(
                top_cities, x="City", y="Visits",
                color="Visits", color_continuous_scale=["#90caf9", "#1a2b4c"],
            )
            fig1.update_traces(marker_line_width=0)
            fig1.update_layout(
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="Segoe UI", color="#1a2b4c", size=11),
                margin=dict(l=40, r=20, t=10, b=50), coloraxis_showscale=False, height=300,
            )
            fig1.update_xaxes(showgrid=False, title_text="", tickangle=-30)
            fig1.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)", title_text="Visits")

            # visit mode pie
            fig2 = px.pie(
                names=mode_counts.index, values=mode_counts.values,
                color_discrete_sequence=["#1a2b4c", "#4fc3f7", "#f5a623",
                                          "#0288d1", "#90caf9", "#e67e22"],
                hole=0.4,
            )
            fig2.update_traces(textfont_size=11, marker=dict(line=dict(color="white", width=2)))
            fig2.update_layout(
                paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                font=dict(family="Segoe UI", color="#1a2b4c", size=11),
                margin=dict(l=10, r=10, t=10, b=10), height=300,
            )

            cfg = {"responsive": True, "displayModeBar": False}

            result = {
                "country": selected_country,
                "total_visits": int(len(df)),
                "avg_rating": round(float(df["Rating"].mean()), 2),
                "stars": stars_html(df["Rating"].mean()),
                "total_cities": int(df["City"].nunique()),
                "total_attractions": int(df["Attraction"].nunique()),
                "top_attractions": top_attractions.to_dict(orient="records"),
                "top_mode": mode_counts.idxmax() if not mode_counts.empty else "N/A",
                "city_chart": pio.to_html(fig1, full_html=False, include_plotlyjs=False, config=cfg),
                "mode_chart": pio.to_html(fig2, full_html=False, include_plotlyjs=False, config=cfg),
            }

    return render_template(
        "country_explorer.html",
        countries=COUNTRIES,
        selected_country=selected_country,
        result=result,
    )


    # ---------------------------------------------------------------
# USER PROFILE (favourites stored client-side via localStorage)
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# USER PROFILE (favourites stored client-side via localStorage)
# ---------------------------------------------------------------
@app.route("/profile")
def profile():
    return render_template("profile.html", all_attractions=ALL_ATTRACTIONS)

# ---------------------------------------------------------------
# ANALYTICS DASHBOARD
# ---------------------------------------------------------------
@app.route("/dashboard")
def dashboard():

    COLORS = ["#1a2b4c", "#2d4a7a", "#4fc3f7", "#0288d1", "#90caf9",
              "#f5a623", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    common_layout = dict(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="Segoe UI", color="#1a2b4c", size=12),
        margin=dict(l=40, r=20, t=44, b=60),
        title_font=dict(size=14, color="#1a2b4c", family="Segoe UI"),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", borderwidth=0),
        coloraxis_showscale=False,
        autosize=True,
        height=380,
    )

    # 1. Top 10 Attractions
    top_attractions = (
        transactions_df["Attraction"].value_counts().head(10).reset_index()
    )
    top_attractions.columns = ["Attraction", "Visits"]
    fig1 = px.bar(
        top_attractions, x="Attraction", y="Visits",
        title="🏛 Top 10 Popular Attractions",
        color="Visits",
        color_continuous_scale=["#90caf9", "#1a2b4c"],
    )
    fig1.update_traces(marker_line_width=0)
    fig1.update_layout(**common_layout)
    fig1.update_xaxes(showgrid=False, tickangle=-35,
                      tickfont=dict(size=10), title_text="")
    fig1.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                      title_text="Visits")

    # 2. Top Countries
    top_countries = transactions_df["Country"].value_counts().reset_index()
    top_countries.columns = ["Country", "Visits"]
    fig2 = px.bar(
        top_countries, x="Country", y="Visits",
        title="🌍 Visits by Country",
        color="Visits",
        color_continuous_scale=["#4fc3f7", "#1a2b4c"],
    )
    fig2.update_traces(marker_line_width=0)
    fig2.update_layout(**common_layout)
    fig2.update_xaxes(showgrid=False, tickangle=-35,
                      tickfont=dict(size=10), title_text="")
    fig2.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                      title_text="Visits")

    # 3. Top Cities
    top_cities = transactions_df["City"].value_counts().head(10).reset_index()
    top_cities.columns = ["City", "Visits"]
    fig3 = px.bar(
        top_cities, x="City", y="Visits",
        title="🏙 Top 10 Cities by Visits",
        color="Visits",
        color_continuous_scale=["#90caf9", "#0288d1"],
    )
    fig3.update_traces(marker_line_width=0)
    fig3.update_layout(**common_layout)
    fig3.update_xaxes(showgrid=False, tickangle=-35,
                      tickfont=dict(size=10), title_text="")
    fig3.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                      title_text="Visits")

    # 4. Ratings Distribution
    fig4 = px.histogram(
        transactions_df, x="Rating", nbins=20,
        title="⭐ Ratings Distribution",
        color_discrete_sequence=["#4fc3f7"],
    )
    fig4.update_traces(marker_line_color="#1a2b4c", marker_line_width=1)
    fig4.update_layout(**common_layout)
    fig4.update_xaxes(showgrid=False, title_text="Rating")
    fig4.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                      title_text="Count")

    # 5. Visit Mode Pie
    mode_counts = transactions_df["VisitMode"].value_counts().reset_index()
    mode_counts.columns = ["VisitMode", "Count"]
    fig5 = px.pie(
        mode_counts, names="VisitMode", values="Count",
        title="🚗 Visit Mode Distribution",
        color_discrete_sequence=["#1a2b4c", "#4fc3f7", "#f5a623",
                                  "#0288d1", "#90caf9", "#e67e22"],
        hole=0.38,
    )
    fig5.update_traces(
        textfont_size=12,
        marker=dict(line=dict(color="white", width=2)),
        pull=[0.04] * len(mode_counts),
    )
    fig5.update_layout(**common_layout)

    # 6. Monthly Trends
    monthly = (
        transactions_df.groupby("VisitMonth").size().reset_index(name="Visits")
    )
    fig6 = px.line(
        monthly, x="VisitMonth", y="Visits", markers=True,
        title="📅 Monthly Tourism Trends",
        color_discrete_sequence=["#1a2b4c"],
    )
    fig6.update_traces(
        line=dict(width=3),
        marker=dict(size=9, color="#4fc3f7",
                    line=dict(width=2, color="#1a2b4c")),
    )
    fig6.update_layout(**common_layout)
    fig6.update_xaxes(
        showgrid=False, title_text="Month",
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"],
    )
    fig6.update_yaxes(showgrid=True, gridcolor="rgba(26,43,76,0.08)",
                      title_text="Visits")

    _cfg = {"responsive": True, "displayModeBar": False}

    charts = {
        "fig1": pio.to_html(fig1, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig1"),
        "fig2": pio.to_html(fig2, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig2"),
        "fig3": pio.to_html(fig3, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig3"),
        "fig4": pio.to_html(fig4, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig4"),
        "fig5": pio.to_html(fig5, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig5"),
        "fig6": pio.to_html(fig6, full_html=False, include_plotlyjs=False,
                            config=_cfg, div_id="fig6"),
    }

    kpi = {
        "total_visits": int(len(transactions_df)),
        "avg_rating": round(float(transactions_df["Rating"].mean()), 2),
        "top_country": top_countries.iloc[0]["Country"],
        "top_attraction": top_attractions.iloc[0]["Attraction"],
    }

    return render_template("dashboard.html", charts=charts, kpi=kpi)


# ---------------------------------------------------------------
# INDIA MAP VISUALIZATION
# ---------------------------------------------------------------
# ---------------------------------------------------------------
# INDIA MAP VISUALIZATION
# ---------------------------------------------------------------
@app.route("/indiamap")
def india_map():
    import folium
    import json

    india_df = transactions_df[
        transactions_df["Country"].str.strip().str.lower() == "india"
    ]

    city_visits = india_df["City"].value_counts().reset_index()
    city_visits.columns = ["City", "Visits"]

    city_visits["Lat"] = city_visits["City"].map(
        lambda c: INDIA_CITY_COORDS.get(c, (None, None))[0]
    )
    city_visits["Lon"] = city_visits["City"].map(
        lambda c: INDIA_CITY_COORDS.get(c, (None, None))[1]
    )

    mapped_df       = city_visits.dropna(subset=["Lat", "Lon"])
    unmapped_cities = city_visits[city_visits["Lat"].isna()]["City"].tolist()

    # Cities from INDIA_CITY_COORDS that have NO visit data at all
    visited_city_names = set(mapped_df["City"])
    reference_cities = {
        name: coords for name, coords in INDIA_CITY_COORDS.items()
        if name not in visited_city_names
    }

    # ── build folium map ──
    m = folium.Map(
        location=[22.5, 82.0],
        zoom_start=5,
        tiles=None,          # no default tile — we add a clean one below
        prefer_canvas=True,
    )

    # clean light tile
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="CartoDB",
        name="CartoDB Light",
        control=False,
    ).add_to(m)

    # GeoJSON state boundaries
    geojson_path = os.path.join("static", "india_states.geojson")
    if os.path.exists(geojson_path):
        with open(geojson_path, encoding="utf-8") as f:
            geo_data = json.load(f)

        folium.GeoJson(
            geo_data,
            name="States",
            style_function=lambda feat: {
                "fillColor":   "#b8d98d",
                "color":       "#ffffff",
                "weight":      1.5,
                "fillOpacity": 0.65,
            },
            highlight_function=lambda feat: {
                "fillColor":   "#4fc3f7",
                "color":       "#1a2b4c",
                "weight":      2.5,
                "fillOpacity": 0.85,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["NAME_1"],
                aliases=["State:"],
                style=(
                    "background-color:white; color:#1a2b4c;"
                    "font-family:Segoe UI; font-size:13px;"
                    "font-weight:bold; padding:6px 10px;"
                    "border-radius:6px; border:1px solid #ccc;"
                ),
            ),
        ).add_to(m)

    # ── city markers WITH visit data ──
    max_visits = mapped_df["Visits"].max() if not mapped_df.empty else 1

    for _, row in mapped_df.iterrows():
        radius = 6 + (row["Visits"] / max_visits) * 18
        folium.CircleMarker(
            location=[row["Lat"], row["Lon"]],
            radius=radius,
            color="#1a2b4c",
            weight=2,
            fill=True,
            fill_color="#f5a623",
            fill_opacity=0.85,
            tooltip=folium.Tooltip(
                f"<b style='color:#1a2b4c;font-size:13px'>{row['City']}</b>"
                f"<br><span style='color:#4fc3f7;font-weight:600'>"
                f"Visits: {row['Visits']}</span>",
                sticky=True,
            ),
        ).add_to(m)

        # bold city label
        folium.Marker(
            location=[row["Lat"] + 0.4, row["Lon"]],
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    'font-family:Segoe UI,sans-serif;'
                    'font-size:11px;'
                    'font-weight:800;'
                    'color:#1a2b4c;'
                    'white-space:nowrap;'
                    'text-shadow:1px 1px 2px white,-1px -1px 2px white,'
                    '1px -1px 2px white,-1px 1px 2px white;'
                    f'">{row["City"]}</div>'
                ),
                icon_size=(120, 20),
                icon_anchor=(60, 10),
            ),
        ).add_to(m)

    # ── reference markers for ALL OTHER cities in INDIA_CITY_COORDS ──
    # (no visit data, but we still want their names shown on the map)
    for name, (lat, lon) in reference_cities.items():
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="#90a4ae",
            weight=1.5,
            fill=True,
            fill_color="#cfd8dc",
            fill_opacity=0.8,
            tooltip=folium.Tooltip(
                f"<b style='color:#1a2b4c;font-size:12px'>{name}</b>"
                f"<br><span style='color:#90a4ae;font-weight:600'>No visits recorded</span>",
                sticky=True,
            ),
        ).add_to(m)

        folium.Marker(
            location=[lat + 0.35, lon],
            icon=folium.DivIcon(
                html=(
                    '<div style="'
                    'font-family:Segoe UI,sans-serif;'
                    'font-size:9.5px;'
                    'font-weight:600;'
                    'color:#5a6b7d;'
                    'white-space:nowrap;'
                    'text-shadow:1px 1px 2px white,-1px -1px 2px white,'
                    '1px -1px 2px white,-1px 1px 2px white;'
                    f'">{name}</div>'
                ),
                icon_size=(110, 18),
                icon_anchor=(55, 9),
            ),
        ).add_to(m)

    # fit bounds to India
    m.fit_bounds([[6, 66], [38, 100]])

    india_map_html = m.get_root().render()

    return render_template(
        "india_map.html",
        india_map_html=india_map_html,
        unmapped_cities=unmapped_cities,
        total_india_visits=int(city_visits["Visits"].sum()) if not city_visits.empty else 0,
        city_table=mapped_df.to_dict(orient="records"),
    )

# ---------------------------------------------------------------
# ABOUT
# ---------------------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)