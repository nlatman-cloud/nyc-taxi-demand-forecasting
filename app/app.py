from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="NYC Taxi Demand Forecasting",
    page_icon="🚕",
    layout="wide",
)


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
APP_DATA_DIR = PROJECT_ROOT / "data" / "app"


# ============================================================
# Data Loading
# ============================================================

@st.cache_data
def load_demand_data():
    """Load lightweight datasets used by the Streamlit application."""

    hourly = pd.read_parquet(
        APP_DATA_DIR / "zone_hourly_patterns.parquet"
    )

    weekday = pd.read_parquet(
        APP_DATA_DIR / "zone_weekday_patterns.parquet"
    )

    zones = pd.read_parquet(
        APP_DATA_DIR / "zone_mapping.parquet"
    )

    december_path = APP_DATA_DIR / "december_predictions.parquet"

    december = (
        pd.read_parquet(december_path)
        if december_path.exists()
        else None
    )

    return hourly, weekday, zones, december


# ============================================================
# Header
# ============================================================

st.title("NYC Taxi Demand Forecasting")

st.markdown(
    """
    **Forecasting hourly Yellow Taxi pickups across 174 NYC taxi zones**

    Built from approximately **48 million 2025 Yellow Taxi trips**, this
    project compares statistical, machine-learning, weather-augmented, and
    deep-learning approaches for one-hour-ahead demand forecasting.
    """
)


# ============================================================
# Headline Metrics
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Trips Processed",
    "48M+"
)

col2.metric(
    "Pickup Zones",
    "174"
)

col3.metric(
    "December MAE",
    "5.72",
    help="Mean absolute error in pickups per zone-hour."
)

col4.metric(
    "MAE Improvement",
    "49.8%",
    help="Improvement over the previous-week persistence baseline."
)

st.divider()


# ============================================================
# Tabs
# ============================================================

overview_tab, demand_tab, model_tab = st.tabs(
    [
        "Overview",
        "Demand Explorer",
        "Model Insights",
    ]
)


# ============================================================
# Overview
# ============================================================

with overview_tab:

    st.header("Forecasting NYC Taxi Demand")

    st.markdown(
        """
        Taxi demand varies substantially across both **location and time**.
        The forecasting task is to predict the number of Yellow Taxi pickups
        in each taxi zone **one hour ahead** using information available
        before the prediction timestamp.

        The project follows the full forecasting workflow from raw trip
        records through feature engineering, chronological validation,
        model comparison, external-factor testing, and final evaluation.
        """
    )

    st.subheader("December 2025 Forecast")

    december_figure = FIGURES_DIR / "december_forecast.png"

    if december_figure.exists():

        st.image(
            december_figure,
            use_container_width=True
        )

    else:

        st.warning(
            "December forecast figure not found."
        )

    st.markdown(
        """
        The final **Random Forest** achieved **5.72 MAE** on held-out
        December data, reducing error by **49.8%** relative to a
        previous-week persistence baseline.

        The model captured the recurring daily demand cycle and broader
        changes across the month, while the largest errors occurred during
        unusual demand peaks and the late-December holiday period.
        """
    )

    st.subheader("Project Pipeline")

    pipe1, pipe2, pipe3, pipe4 = st.columns(4)

    pipe1.markdown(
        """
        **1. Data**

        48M+ Yellow Taxi trips
        """
    )

    pipe2.markdown(
        """
        **2. Features**

        Calendar, lags & rolling demand
        """
    )

    pipe3.markdown(
        """
        **3. Modeling**

        Baselines → Linear → RF → GRU
        """
    )

    pipe4.markdown(
        """
        **4. Evaluation**

        Chronological out-of-time testing
        """
    )


# ============================================================
# Demand Explorer
# ============================================================

with demand_tab:

    st.header("Demand Explorer")

    st.markdown(
        """
        Select a taxi zone to explore its typical demand patterns and
        compare the final Random Forest forecast with observed December
        demand.
        """
    )

    try:

        (
            hourly_patterns,
            weekday_patterns,
            zone_mapping,
            december_predictions,
        ) = load_demand_data()


        # ----------------------------------------------------
        # Zone Selector
        # ----------------------------------------------------

        zone_mapping = zone_mapping.sort_values("Zone")

        zone_name = st.selectbox(
            "Select a Taxi Zone",
            zone_mapping["Zone"].tolist()
        )

        zone_id = zone_mapping.loc[
            zone_mapping["Zone"] == zone_name,
            "PULocationID"
        ].iloc[0]

        st.subheader(zone_name)


        # ----------------------------------------------------
        # Prepare Hourly Demand
        # ----------------------------------------------------

        zone_hourly = (
            hourly_patterns[
                hourly_patterns["PULocationID"] == zone_id
            ]
            .sort_values("hour")
            .set_index("hour")
        )


        # ----------------------------------------------------
        # Prepare Weekday Demand
        # ----------------------------------------------------

        zone_weekday = weekday_patterns[
            weekday_patterns["PULocationID"] == zone_id
        ].copy()

        day_names = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        zone_weekday["day"] = (
            zone_weekday["day_of_week"]
            .map(day_names)
        )

        zone_weekday["day"] = pd.Categorical(
            zone_weekday["day"],
            categories=list(day_names.values()),
            ordered=True
        )

        zone_weekday = (
            zone_weekday
            .sort_values("day")
            .set_index("day")
        )


        # ----------------------------------------------------
        # Historical Demand Patterns
        # ----------------------------------------------------

        st.markdown("### Historical Demand Patterns")

        pattern_col1, pattern_col2 = st.columns(2)

        with pattern_col1:

            st.markdown("#### Average Demand by Hour")

            st.line_chart(
                zone_hourly["average_demand"],
                x_label="Hour of Day",
                y_label="Average Pickups",
                height=350
            )

        with pattern_col2:

            st.markdown("#### Average Demand by Day")

            st.bar_chart(
                zone_weekday["average_demand"],
                x_label="Day",
                y_label="Average Pickups",
                height=350
            )


        # ----------------------------------------------------
        # December Forecast
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            "### December 2025: Actual vs. Predicted"
        )

        st.markdown(
            """
            Compare the final Random Forest forecast with observed demand
            for the selected zone throughout the held-out December
            evaluation period.
            """
        )

        if december_predictions is None:

            st.info(
                "December zone-level predictions have not been exported yet."
            )

        else:

            zone_december = (
                december_predictions[
                    december_predictions["PULocationID"] == zone_id
                ]
                .sort_values("pickup_hour")
                [
                    [
                        "pickup_hour",
                        "trip_count",
                        "prediction",
                    ]
                ]
                .rename(
                    columns={
                        "trip_count": "Actual",
                        "prediction": "Predicted",
                    }
                )
                .set_index("pickup_hour")
            )

            if zone_december.empty:

                st.info(
                    "No December forecast data are available for this zone."
                )

            else:

                st.line_chart(
                    zone_december,
                    x_label="Date",
                    y_label="Pickups",
                    height=450
                )

                zone_mae = (
                    zone_december["Actual"]
                    - zone_december["Predicted"]
                ).abs().mean()

                actual_total = zone_december["Actual"].sum()
                predicted_total = zone_december["Predicted"].sum()

                forecast_col1, forecast_col2, forecast_col3 = (
                    st.columns(3)
                )

                forecast_col1.metric(
                    "December Zone MAE",
                    f"{zone_mae:.2f}",
                    help="Mean absolute error for the selected zone during December."
                )

                forecast_col2.metric(
                    "Actual Pickups",
                    f"{int(round(actual_total)):,}"
                )

                forecast_col3.metric(
                    "Predicted Pickups",
                    f"{int(round(predicted_total)):,}"
                )


    except FileNotFoundError:

        st.warning(
            """
            Streamlit demand datasets were not found.

            Run the Streamlit export section at the end of
            Notebook 02 before using the Demand Explorer.
            """
        )


# ============================================================
# Model Insights
# ============================================================

with model_tab:

    st.header("Model Insights")

    st.markdown(
        """
        Model selection was based on **chronological validation performance**,
        not model complexity. Simpler baselines, traditional machine-learning
        models, weather augmentation, and a neural sequence model were all
        evaluated before selecting the final specification.
        """
    )


    # --------------------------------------------------------
    # Model Comparison
    # --------------------------------------------------------

    st.subheader("Validation Performance")

    model_figure = FIGURES_DIR / "model_comparison.png"

    if model_figure.exists():

        st.image(
            model_figure,
            use_container_width=True
        )

    else:

        st.warning(
            "Model comparison figure not found."
        )

    st.markdown(
        """
        The **Random Forest** achieved the lowest validation MAE.

        The GRU successfully learned temporal structure directly from the
        previous 168 hours of demand, but its additional complexity did not
        outperform the feature-engineered Random Forest.
        """
    )


    # --------------------------------------------------------
    # Weather Experiment
    # --------------------------------------------------------

    st.divider()

    st.subheader("Weather Experiment")

    st.markdown(
        """
        Weather was investigated both descriptively and as a potential
        forecasting input. Predictive testing used lagged weather variables
        so that the experiment did not rely on information unavailable at
        forecast time.
        """
    )

    weather_col1, weather_col2 = st.columns(
        [2, 1]
    )

    with weather_col1:

        rainfall_figure = (
            FIGURES_DIR / "rainfall_demand.png"
        )

        if rainfall_figure.exists():

            st.image(
                rainfall_figure,
                use_container_width=True
            )

        else:

            st.warning(
                "Rainfall demand figure not found."
            )

    with weather_col2:

        st.metric(
            "Heavy-Rain Demand Deviation",
            "+13.0%",
            help="Median demand deviation from the typical day-of-week and hour pattern."
        )

        st.metric(
            "Backtest Months Improved",
            "0 / 5"
        )

        st.markdown(
            """
            Heavy rainfall was associated with substantially higher taxi
            demand, but leakage-safe lagged weather features **did not improve
            forecasting performance** across the five expanding-window
            backtests.

            This highlights the distinction between an explanatory
            relationship and a feature that adds predictive value.
            """
        )


    # --------------------------------------------------------
    # Final Model Selection
    # --------------------------------------------------------

    st.divider()

    st.subheader("Why Random Forest?")

    reason1, reason2 = st.columns(2)

    with reason1:

        st.markdown(
            """
            **Performance**
            
            - Lowest validation MAE among tested models
            - 49.8% lower December MAE than weekly persistence
            - No negative demand predictions
            """
        )

    with reason2:

        st.markdown(
            """
            **Model Selection**
            
            - Weather augmentation did not improve forecasts
            - GRU complexity did not outperform Random Forest
            - December was not used for original model selection
            """
        )

    st.markdown(
        """
        The final model was therefore selected based on **out-of-time
        forecasting performance rather than model complexity**.
        """
    )