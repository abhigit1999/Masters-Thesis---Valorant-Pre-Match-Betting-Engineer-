import streamlit as st
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pickle
import io
import matplotlib.pyplot as plt
import numpy as np

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["main_db"]

# Load the superdataset
superdataset = pd.DataFrame(list(db["superdataset1"].find()))
if "_id" in superdataset.columns:
    superdataset = superdataset.drop(columns=["_id"])

# Extract unique values for dropdowns
team_ids = sorted(superdataset['Team_A_ID'].unique().tolist())
map_names = sorted(superdataset['Map'].unique().tolist())

# Features and target column
features = [
    'Map', 
    'Team_A_ID', 'Team_A_Kills', 'Team_A_Deaths', 'Team_A_ACS', 
    'Team_A_Headshot_Percentage', 'Team_A_First_Bloods', 'Team_A_Assists',
    'Team_B_ID', 'Team_B_Kills', 'Team_B_Deaths', 'Team_B_ACS', 
    'Team_B_Headshot_Percentage', 'Team_B_First_Bloods', 'Team_B_Assists'
]
target = 'Winner'

# Load model, scaler, and label encoder from MongoDB
model_data = db["models"].find_one({"model_name": "xgboost"})
model_buffer = io.BytesIO(model_data["model_data"])
saved_xgb_model = joblib.load(model_buffer)
label_encoder = pickle.loads(model_data["label_encoder"])
scaler = pickle.loads(model_data["scaler"])


def predict_match_outcome(team_a_id, team_b_id, map_name):
    """
    Predict the winning probabilities for a Valorant match using an XGBoost model.
    """
    if team_a_id not in superdataset['Team_A_ID'].values:
        raise ValueError(f"Team A ID {team_a_id} is not valid.")
    if team_b_id not in superdataset['Team_B_ID'].values:
        raise ValueError(f"Team B ID {team_b_id} is not valid.")
    if team_a_id == team_b_id:
        raise ValueError("Team A ID and Team B ID cannot be the same.")
    if map_name not in label_encoder.classes_:
        raise ValueError(f"Map name '{map_name}' not recognized by the label encoder.")

    # Extract median stats for each team
    def median_or_zero(df, col, filter_col, val):
        med = df[df[filter_col] == val][col].median()
        return 0 if pd.isna(med) else med

    input_data = {
        "Map": label_encoder.transform([map_name])[0],
        "Team_A_ID": team_a_id,
        "Team_A_Kills": median_or_zero(superdataset, "Team_A_Kills", "Team_A_ID", team_a_id),
        "Team_A_Deaths": median_or_zero(superdataset, "Team_A_Deaths", "Team_A_ID", team_a_id),
        "Team_A_ACS": median_or_zero(superdataset, "Team_A_ACS", "Team_A_ID", team_a_id),
        "Team_A_Headshot_Percentage": median_or_zero(superdataset, "Team_A_Headshot_Percentage", "Team_A_ID", team_a_id),
        "Team_A_First_Bloods": median_or_zero(superdataset, "Team_A_First_Bloods", "Team_A_ID", team_a_id),
        "Team_A_Assists": median_or_zero(superdataset, "Team_A_Assists", "Team_A_ID", team_a_id),
        "Team_B_ID": team_b_id,
        "Team_B_Kills": median_or_zero(superdataset, "Team_B_Kills", "Team_B_ID", team_b_id),
        "Team_B_Deaths": median_or_zero(superdataset, "Team_B_Deaths", "Team_B_ID", team_b_id),
        "Team_B_ACS": median_or_zero(superdataset, "Team_B_ACS", "Team_B_ID", team_b_id),
        "Team_B_Headshot_Percentage": median_or_zero(superdataset, "Team_B_Headshot_Percentage", "Team_B_ID", team_b_id),
        "Team_B_First_Bloods": median_or_zero(superdataset, "Team_B_First_Bloods", "Team_B_ID", team_b_id),
        "Team_B_Assists": median_or_zero(superdataset, "Team_B_Assists", "Team_B_ID", team_b_id)
    }

    input_df = pd.DataFrame([input_data]).fillna(0)
    input_df = input_df[features]
    input_scaled = scaler.transform(input_df)
    probabilities = saved_xgb_model.predict_proba(input_scaled)
    prob_a = probabilities[0][1]  # Probability that Team A wins
    prob_b = probabilities[0][0]  # Probability that Team B wins

    # Check for probability sum
    sum_probs = prob_a + prob_b
    if not (0.999 <= sum_probs <= 1.001):
        # Normalize if needed
        prob_a /= sum_probs
        prob_b /= sum_probs

    return prob_a, prob_b


def calculate_odds_and_earnings(prob_a, prob_b, bet_amount, margin=0.05):
    """
    Convert probabilities into realistic bookmaker odds and then calculate potential earnings.
    
    Parameters:
    - prob_a (float): Probability of Team A winning.
    - prob_b (float): Probability of Team B winning.
    - bet_amount (float): Amount of money bet.
    - margin (float): Overround margin the bookmaker adds (default 5%).

    Returns:
    - odds_a (float): Adjusted odds for Team A.
    - odds_b (float): Adjusted odds for Team B.
    - earnings_a (float): Potential payout if Team A wins.
    - earnings_b (float): Potential payout if Team B wins.
    """
    if prob_a <= 0 or prob_b <= 0:
        raise ValueError("Probabilities must be greater than 0.")

    # Fair odds
    fair_odds_a = 1.0 / prob_a
    fair_odds_b = 1.0 / prob_b

    # Implied probabilities
    implied_a = 1.0 / fair_odds_a
    implied_b = 1.0 / fair_odds_b
    sum_implied = implied_a + implied_b

    # Apply the margin (overround)
    desired_sum = 1 + margin
    scale_factor = desired_sum / sum_implied
    adj_prob_a = implied_a * scale_factor
    adj_prob_b = implied_b * scale_factor

    # Recalculate odds with margin
    odds_a = 1.0 / adj_prob_a
    odds_b = 1.0 / adj_prob_b

    # Potential earnings (includes stake)
    earnings_a = bet_amount * odds_a
    earnings_b = bet_amount * odds_b

    return odds_a, odds_b, earnings_a, earnings_b


def plot_probabilities(team_a_label, team_b_label, prob_a, prob_b):
    """
    Plot winning probabilities for Team A and Team B with improved styling.
    """
    labels = [team_a_label, team_b_label]
    values = [prob_a, prob_b]

    # Enhanced color scheme
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for Team A, Orange for Team B

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", height=0.5)

    # Aesthetic improvements
    ax.set_xlim(0, 1)
    ax.set_facecolor("#f0f0f0")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.2))
    ax.xaxis.set_tick_params(color='#333333')
    ax.yaxis.set_ticks_position('none')
    ax.grid(axis='x', linestyle='--', color='#cccccc', alpha=0.7)

    # Label bars with percentages
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.02, 
            bar.get_y() + bar.get_height() / 2, 
            f"{val * 100:.1f}%", 
            va="center", 
            ha="left", 
            fontsize=12, 
            color="#333333", 
            weight="bold"
        )

    # Title and styling
    ax.set_title(
        "Winning Probabilities", 
        fontsize=16, 
        fontweight="bold", 
        color="#333333", 
        pad=15
    )

    plt.tight_layout()
    st.pyplot(fig)


# Streamlit UI
st.title("VALORANT Pre-Match Predictor")
st.subheader(" Valorant Pre-Match Betting Advisory System")

team_a_id = st.selectbox("(Attackers) Team A ID", team_ids)
team_b_id = st.selectbox("(Defenders) Team B ID", team_ids)
map_name = st.selectbox("Select Map Name", map_names)
bet_amount = st.number_input("Enter Bet Amount (In Euros)", min_value=0.0, step=0.1)

if st.button("Predict"):
    try:
        # Predict probabilities using the model
        prob_a, prob_b = predict_match_outcome(team_a_id, team_b_id, map_name)

        # Visualize probabilities
        st.subheader("Winning Probabilities Visualization")
        plot_probabilities("Team A", "Team B", prob_a, prob_b)

        # Calculate realistic odds and earnings
        odds_a, odds_b, earnings_a, earnings_b = calculate_odds_and_earnings(prob_a, prob_b, bet_amount, margin=0.05)

        # Display results
        st.success(f"Winning Probabilities - Team A: {prob_a * 100:.2f}%, Team B: {prob_b * 100:.2f}%")
        st.info(f"Realistic Odds (with margin) - Team A: {odds_a:.2f}, Team B: {odds_b:.2f}")
        st.success(f"Potential Payout - Team A: €{earnings_a:.2f}, Team B: €{earnings_b:.2f}")
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
