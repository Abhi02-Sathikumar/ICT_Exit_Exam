import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Configuration
# =========================================================

MODEL_FILE = Path("pokemon_type_prediction_bundle.pkl")


# =========================================================
# Load saved model bundle
# =========================================================

@st.cache_resource
def load_model_bundle():
    with open(MODEL_FILE, "rb") as file:
        return pickle.load(file)


try:
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error(
        f"Could not find {MODEL_FILE}. "
        "Place the pickle file in the same folder as app.py."
    )
    st.stop()

fitted_models = bundle["fitted_models"]
best_model_name = bundle["best_model_name"]
best_model = bundle["best_model"]

feature_columns = bundle["feature_columns"]
numeric_columns = bundle.get(
    "numeric_columns",
    [
        "height_dm",
        "weight_hg",
        "base_experience",
        "hp",
        "attack",
        "defense",
    ],
)

outlier_bounds = bundle.get("outlier_bounds", {})
base_experience_median = bundle.get("base_experience_median", 170.0)
target_classes = bundle.get("target_classes", sorted(fitted_models.keys()))


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="Pokémon Type Predictor",
    page_icon="⚡",
    layout="wide",
)


# =========================================================
# Helper functions
# =========================================================

def preprocess_input(input_values):
    """
    Reproduce the preprocessing used during training:
    - Fill missing type_2 with 'none'
    - Fill missing base experience with the training median
    - Clip numerical values using saved IQR bounds
    - One-hot encode type_2
    - Match the training feature-column order
    """

    row = pd.DataFrame([input_values])

    # Fill missing values
    row["type_2"] = row["type_2"].fillna("none")
    row["base_experience"] = row["base_experience"].fillna(
        base_experience_median
    )

    # Apply saved outlier limits
    for column in numeric_columns:
        if column in row.columns and column in outlier_bounds:
            lower = outlier_bounds[column]["lower"]
            upper = outlier_bounds[column]["upper"]

            row[column] = row[column].clip(
                lower=lower,
                upper=upper
            )

    # One-hot encode type_2 exactly as done during training
    row = pd.get_dummies(
        row,
        columns=["type_2"],
        dtype=int
    )

    # Add any missing training columns and remove unexpected columns
    row = row.reindex(columns=feature_columns, fill_value=0)

    # Match the numeric format used by scikit-learn
    row = row.astype(float)

    return row


def make_prediction(input_values, model_name):
    processed_input = preprocess_input(input_values)

    model = fitted_models[model_name]

    prediction = model.predict(processed_input)[0]

    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(processed_input)[0]

    return prediction, probabilities


# =========================================================
# Sidebar navigation
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page:",
    [
        "Home",
        "Prediction",
        "Result",
    ],
)


# =========================================================
# Home page
# =========================================================

if page == "Home":
    st.title("⚡ Pokémon Primary Type Predictor")

    st.subheader("Welcome")

    st.write(
        """
        This application predicts a Pokémon's primary type (`type_1`)
        using physical and statistical features such as height, weight,
        HP, attack, defense, base experience, and secondary type.
        """
    )

    st.info(
        f"The best saved model is: **{best_model_name}**"
    )

    st.subheader("Available models")

    for model_name in fitted_models:
        st.write(f"• {model_name}")

    st.subheader("Input features")

    st.write(
        """
        - Height
        - Weight
        - Base experience
        - Secondary type
        - HP
        - Attack
        - Defense
        """
    )

    st.warning(
        "The prediction is based on the preprocessing used during model training."
    )


# =========================================================
# Prediction page
# =========================================================

elif page == "Prediction":
    st.title("🔮 Make a Prediction")

    st.write(
        "Enter the Pokémon characteristics below."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            height_dm = st.number_input(
                "Height (dm)",
                min_value=0.0,
                value=10.0,
                step=0.1,
            )

            weight_hg = st.number_input(
                "Weight (hg)",
                min_value=0.0,
                value=100.0,
                step=0.1,
            )

            base_experience = st.number_input(
                "Base Experience",
                min_value=0.0,
                value=float(base_experience_median),
                step=1.0,
            )

            hp = st.number_input(
                "HP",
                min_value=0.0,
                value=60.0,
                step=1.0,
            )

        with col2:
            attack = st.number_input(
                "Attack",
                min_value=0.0,
                value=70.0,
                step=1.0,
            )

            defense = st.number_input(
                "Defense",
                min_value=0.0,
                value=70.0,
                step=1.0,
            )

            type_2_options = [
                "none",
                "bug",
                "dark",
                "dragon",
                "electric",
                "fairy",
                "fighting",
                "fire",
                "flying",
                "ghost",
                "grass",
                "ground",
                "ice",
                "normal",
                "poison",
                "psychic",
                "rock",
                "steel",
                "water",
            ]

            type_2 = st.selectbox(
                "Secondary Type",
                type_2_options,
            )

            selected_model = st.selectbox(
                "Classification Model",
                list(fitted_models.keys()),
                index=list(fitted_models.keys()).index(
                    best_model_name
                ),
            )

        submitted = st.form_submit_button(
            "Predict Primary Type"
        )

    if submitted:
        input_values = {
            "height_dm": height_dm,
            "weight_hg": weight_hg,
            "base_experience": base_experience,
            "type_2": type_2,
            "hp": hp,
            "attack": attack,
            "defense": defense,
        }

        try:
            prediction, probabilities = make_prediction(
                input_values,
                selected_model,
            )

            st.session_state["prediction"] = prediction
            st.session_state["probabilities"] = probabilities
            st.session_state["input_values"] = input_values
            st.session_state["selected_model"] = selected_model

            st.success(
                "Prediction completed. Open the Result page "
                "to view the result."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# =========================================================
# Result page
# =========================================================

elif page == "Result":
    st.title("📊 Prediction Result")

    if "prediction" not in st.session_state:
        st.info(
            "No prediction is available yet. "
            "Go to the Prediction page first."
        )
        st.stop()

    prediction = st.session_state["prediction"]
    probabilities = st.session_state.get("probabilities")
    input_values = st.session_state["input_values"]
    selected_model = st.session_state["selected_model"]

    st.success(
        f"Predicted primary type: **{str(prediction).upper()}**"
    )

    st.write(f"Model used: **{selected_model}**")

    st.subheader("Input values")

    input_table = pd.DataFrame(
        {
            "Feature": list(input_values.keys()),
            "Value": list(input_values.values()),
        }
    )

    st.table(input_table)

    if probabilities is not None:
        st.subheader("Prediction probabilities")

        probability_table = pd.DataFrame(
            {
                "Type": fitted_models[selected_model]
                .classes_,
                "Probability": probabilities,
            }
        ).sort_values(
            "Probability",
            ascending=False,
        )

        probability_table["Probability"] = (
            probability_table["Probability"] * 100
        ).round(2)

        probability_table = probability_table.rename(
            columns={
                "Probability": "Probability (%)"
            }
        )

        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            probability_table.set_index("Type")["Probability (%)"]
        )