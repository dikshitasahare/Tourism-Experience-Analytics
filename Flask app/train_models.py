"""
train_models.py
Generates synthetic tourism data and trains:
 - Regression model -> predicts Rating
 - Classification model -> predicts VisitMode
 - Item-based collaborative filtering similarity matrix -> recommendations

Run this once: python train_models.py
It creates /data and /models folders used by app.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import joblib
import random

random.seed(42)
np.random.seed(42)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ---------------------------------------------------------------
# 1. Build reference data: countries -> cities -> attractions
# ---------------------------------------------------------------
geo = {
    "India": {
        "Goa": ["Goa Beach", "Fort Aguada", "Dudhsagar Falls"],
        "Mumbai": ["Gateway of India", "Marine Drive", "Elephanta Caves"],
        "Chennai": ["Marina Beach", "Kapaleeshwarar Temple"],
        "Kerala": ["Kovalam Beach", "Alleppey Backwaters", "Munnar Hills"],
        "Puri": ["Puri Beach", "Jagannath Temple"],
    },
    "USA": {
        "New York": ["Statue of Liberty", "Central Park", "Times Square"],
        "Florida": ["Miami Beach", "Universal Studios"],
        "California": ["Golden Gate Bridge", "Hollywood Walk of Fame"],
    },
    "France": {
        "Paris": ["Eiffel Tower", "Louvre Museum", "Notre Dame"],
        "Nice": ["Promenade des Anglais"],
    },
    "Thailand": {
        "Phuket": ["Patong Beach", "Big Buddha"],
        "Bangkok": ["Grand Palace", "Chatuchak Market"],
    },
    "Australia": {
        "Sydney": ["Sydney Opera House", "Bondi Beach"],
        "Gold Coast": ["Surfers Paradise", "Diu Beach"],  # demo name kept for variety
    },
}

attraction_types = ["Beach", "Historical Site", "Religious Site", "Park",
                     "Museum", "Hill Station", "Market", "Landmark"]

rows_attr = []
attr_id = 1
for country, cities in geo.items():
    for city, attractions in cities.items():
        for name in attractions:
            rows_attr.append({
                "AttractionId": attr_id,
                "Attraction": name,
                "City": city,
                "Country": country,
                "AttractionType": random.choice(attraction_types)
            })
            attr_id += 1

attractions_df = pd.DataFrame(rows_attr)
attractions_df.to_csv("data/attractions.csv", index=False)

visit_modes = ["Business", "Family", "Couples", "Friends", "Solo"]
months = list(range(1, 13))

# ---------------------------------------------------------------
# 2. Generate synthetic transactions (UserId, Attraction, Month, VisitMode, Rating)
# ---------------------------------------------------------------
n_users = 800
n_transactions = 6000

rows_tx = []
for i in range(n_transactions):
    user_id = random.randint(1, n_users)
    attr_row = attractions_df.sample(1).iloc[0]
    month = random.choice(months)
    mode = random.choice(visit_modes)

    # base rating influenced by attraction type + a little randomness, kept 1-5
    base = {
        "Beach": 4.3, "Historical Site": 4.0, "Religious Site": 4.1,
        "Park": 3.9, "Museum": 3.8, "Hill Station": 4.4,
        "Market": 3.6, "Landmark": 4.2
    }[attr_row["AttractionType"]]
    rating = np.clip(np.random.normal(base, 0.6), 1, 5)

    rows_tx.append({
        "UserId": user_id,
        "AttractionId": attr_row["AttractionId"],
        "Attraction": attr_row["Attraction"],
        "City": attr_row["City"],
        "Country": attr_row["Country"],
        "AttractionType": attr_row["AttractionType"],
        "VisitMonth": month,
        "VisitMode": mode,
        "Rating": round(rating, 1)
    })

transactions_df = pd.DataFrame(rows_tx)
transactions_df.to_csv("data/transactions.csv", index=False)

print(f"Generated {len(attractions_df)} attractions and {len(transactions_df)} transactions.")

# ---------------------------------------------------------------
# 3. Encode categorical columns
# ---------------------------------------------------------------
le_country = LabelEncoder()
le_city = LabelEncoder()
le_attraction = LabelEncoder()
le_visitmode = LabelEncoder()

transactions_df["Country_enc"] = le_country.fit_transform(transactions_df["Country"])
transactions_df["City_enc"] = le_city.fit_transform(transactions_df["City"])
transactions_df["Attraction_enc"] = le_attraction.fit_transform(transactions_df["Attraction"])
transactions_df["VisitMode_enc"] = le_visitmode.fit_transform(transactions_df["VisitMode"])

joblib.dump(le_country, "models/le_country.pkl")
joblib.dump(le_city, "models/le_city.pkl")
joblib.dump(le_attraction, "models/le_attraction.pkl")
joblib.dump(le_visitmode, "models/le_visitmode.pkl")

# ---------------------------------------------------------------
# 4. Regression model -> predict Rating
# ---------------------------------------------------------------
feature_cols_reg = ["Country_enc", "City_enc", "Attraction_enc", "VisitMonth"]
X_reg = transactions_df[feature_cols_reg]
y_reg = transactions_df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
reg_model = RandomForestRegressor(n_estimators=150, random_state=42)
reg_model.fit(X_train, y_train)
print("Regression R2:", r2_score(y_test, reg_model.predict(X_test)))

joblib.dump(reg_model, "models/rating_model.pkl")

# ---------------------------------------------------------------
# 5. Classification model -> predict VisitMode
# ---------------------------------------------------------------
feature_cols_clf = ["Country_enc", "City_enc", "Attraction_enc", "VisitMonth", "Rating"]
X_clf = transactions_df[feature_cols_clf]
y_clf = transactions_df["VisitMode_enc"]

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
clf_model = RandomForestClassifier(n_estimators=150, random_state=42)
clf_model.fit(X_train, y_train)
print("Classification Accuracy:", accuracy_score(y_test, clf_model.predict(X_test)))

joblib.dump(clf_model, "models/visitmode_model.pkl")

# ---------------------------------------------------------------
# 6. Item-based Collaborative Filtering -> recommendation similarity matrix
# ---------------------------------------------------------------
user_item = transactions_df.pivot_table(
    index="UserId", columns="Attraction", values="Rating", aggfunc="mean"
).fillna(0)

item_similarity = cosine_similarity(user_item.T)
item_sim_df = pd.DataFrame(item_similarity, index=user_item.columns, columns=user_item.columns)

joblib.dump(item_sim_df, "models/item_similarity.pkl")

print("\nAll models trained and saved to /models")
print("Sample attractions for recommend page:", list(user_item.columns[:5]))