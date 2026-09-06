import os
import spaces
import joblib
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download


# ---------------------------------------------------
# Hugging Face Model Configuration
# ---------------------------------------------------

DEFAULT_MODEL_REPO = "Ms21063/tourism-purchase-model"

MODEL_REPO = os.getenv(
    "HF_MODEL_REPO",
    DEFAULT_MODEL_REPO
)

MODEL_FILE = "best_model.joblib"


# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

def load_model():
    """Load the registered model from the Hugging Face Model Hub."""

    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            repo_type="model",
        )

        model = joblib.load(model_path)

        return (
            model,
            f"Loaded registered model: {MODEL_REPO}"
        )

    except Exception as exc:

        # Local fallback for local testing
        local_path = os.path.join(
            os.path.dirname(__file__),
            "best_model.joblib"
        )

        if os.path.exists(local_path):

            model = joblib.load(local_path)

            return (
                model,
                "Loaded local fallback model."
            )

        raise RuntimeError(
            f"Unable to load model from {MODEL_REPO}. "
            f"Register the model first. Details: {exc}"
        )


MODEL, MODEL_STATUS = load_model()


# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

@spaces.GPU
def predict(
    age,
    contact,
    city_tier,
    occupation,
    gender,
    persons_visiting,
    property_star,
    marital_status,
    trips,
    passport,
    own_car,
    children_visiting,
    designation,
    monthly_income,
    pitch_score,
    product_pitched,
    followups,
    duration,
):
    """
    Create a customer dataframe and return:
    1. Purchase probability
    2. Marketing recommendation
    """

    row = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": contact,
                "CityTier": city_tier,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": persons_visiting,
                "PreferredPropertyStar": property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": trips,
                "Passport": passport,
                "OwnCar": own_car,
                "NumberOfChildrenVisiting": children_visiting,
                "Designation": designation,
                "MonthlyIncome": monthly_income,
                "PitchSatisfactionScore": pitch_score,
                "ProductPitched": product_pitched,
                "NumberOfFollowups": followups,
                "DurationOfPitch": duration,
            }
        ]
    )

    # Get purchase probability
    probability = float(
        MODEL.predict_proba(row)[:, 1][0]
    )

    # Classification threshold
    prediction = int(probability >= 0.50)

    # Business recommendation
    if prediction:

        decision = (
            "Likely to purchase — "
            "prioritize this customer."
        )

    else:

        decision = (
            "Less likely to purchase — "
            "consider lower-priority outreach."
        )

    return (
        f"{probability:.1%}",
        decision
    )


# ---------------------------------------------------
# Gradio User Interface
# ---------------------------------------------------

with gr.Blocks(
    title="Wellness Tourism Package Predictor"
) as demo:

    gr.Markdown(
        """
        # ✈️ Wellness Tourism Package Predictor

        **Visit with Us** — Predict whether a customer is
        likely to purchase the new Wellness Tourism Package
        before contacting them.
        """
    )

    gr.Markdown(
        f"**Model status:** {MODEL_STATUS}"
    )


    # ---------------------------------------------------
    # Customer Information
    # ---------------------------------------------------

    with gr.Row():

        # -----------------------------------------------
        # Left Column
        # -----------------------------------------------

        with gr.Column():

            age = gr.Number(
                label="Age",
                value=35,
                minimum=18,
                maximum=100,
                precision=0,
            )

            contact = gr.Dropdown(
                [
                    "Company Invited",
                    "Self Enquiry"
                ],
                label="Type of Contact",
                value="Self Enquiry",
            )

            city_tier = gr.Dropdown(
                [1, 2, 3],
                label="City Tier",
                value=1,
            )

            occupation = gr.Dropdown(
                [
                    "Salaried",
                    "Free Lancer",
                    "Small Business",
                    "Large Business"
                ],
                label="Occupation",
                value="Salaried",
            )

            gender = gr.Dropdown(
                [
                    "Female",
                    "Male",
                    "Fe Male"
                ],
                label="Gender",
                value="Male",
            )

            persons_visiting = gr.Number(
                label="Number of Persons Visiting",
                value=2,
                minimum=1,
                maximum=20,
                precision=0,
            )

            property_star = gr.Dropdown(
                [3, 4, 5],
                label="Preferred Property Star",
                value=4,
            )

            marital_status = gr.Dropdown(
                [
                    "Single",
                    "Divorced",
                    "Married",
                    "Unmarried"
                ],
                label="Marital Status",
                value="Married",
            )


        # -----------------------------------------------
        # Right Column
        # -----------------------------------------------

        with gr.Column():

            trips = gr.Number(
                label="Number of Trips",
                value=3,
                minimum=0,
                maximum=20,
                precision=0,
            )

            passport = gr.Dropdown(
                [0, 1],
                label="Passport",
                value=1,
            )

            own_car = gr.Dropdown(
                [0, 1],
                label="Own Car",
                value=1,
            )

            children_visiting = gr.Number(
                label="Number of Children Visiting",
                value=0,
                minimum=0,
                maximum=10,
                precision=0,
            )

            designation = gr.Dropdown(
                [
                    "Executive",
                    "Manager",
                    "Senior Manager",
                    "AVP",
                    "VP"
                ],
                label="Designation",
                value="Executive",
            )

            monthly_income = gr.Number(
                label="Monthly Income",
                value=50000,
                minimum=1000,
                maximum=100000,
            )

            pitch_score = gr.Slider(
                minimum=1,
                maximum=5,
                value=4,
                step=1,
                label="Pitch Satisfaction Score",
            )

            product_pitched = gr.Dropdown(
                [
                    "Basic",
                    "Standard",
                    "Deluxe",
                    "Super Deluxe",
                    "King"
                ],
                label="Product Pitched",
                value="Deluxe",
            )

            followups = gr.Number(
                label="Number of Followups",
                value=4,
                minimum=0,
                maximum=10,
                precision=0,
            )

            duration = gr.Number(
                label="Duration of Pitch (minutes)",
                value=15,
                minimum=1,
                maximum=60,
            )


    # ---------------------------------------------------
    # Prediction Button
    # ---------------------------------------------------

    predict_button = gr.Button(
        "Predict Purchase Probability",
        variant="primary",
    )


    # ---------------------------------------------------
    # Outputs
    # ---------------------------------------------------

    with gr.Row():

        probability_output = gr.Textbox(
            label="Purchase Probability"
        )

        decision_output = gr.Textbox(
            label="Marketing Recommendation"
        )


    # ---------------------------------------------------
    # Button Action
    # ---------------------------------------------------

    predict_button.click(
        fn=predict,

        inputs=[
            age,
            contact,
            city_tier,
            occupation,
            gender,
            persons_visiting,
            property_star,
            marital_status,
            trips,
            passport,
            own_car,
            children_visiting,
            designation,
            monthly_income,
            pitch_score,
            product_pitched,
            followups,
            duration,
        ],

        outputs=[
            probability_output,
            decision_output,
        ],
    )


# ---------------------------------------------------
# Launch Application
# ---------------------------------------------------

if __name__ == "__main__":
    demo.launch()
