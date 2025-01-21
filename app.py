import logging
from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import randint
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variable for MongoDB URI
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://dbUser:12345@cluster0.dgpab.mongodb.net/project11?retryWrites=true&w=majority&tls=true')

def find_amenities(x, places):
    temp = places[places["_id"].isin([ObjectId(id) for id in x])]
    temp = temp[temp["amenities"] != ""]
    temp = temp["amenities"].to_list()
    return " ".join(sorted(temp))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    category = data.get('category')
    days = data.get('days')

    if not user_id or not category or not days:
        return jsonify({"error": "user_id, category, and days are required"}), 400

    logger.info(f"Received user_id: {user_id}")
    logger.info(f"Received category: {category}")
    logger.info(f"Received days: {days}")

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["project11"]
    destinations = db["destinations"]
    saved_destinations = db["saved_destinations"]

    # Fetch all destinations
    data = list(destinations.find({}, {"_id": 1, "category": 1, "amenities": 1}))
    places = pd.DataFrame(data)

    logger.info(f"Fetched all destinations: {places.to_dict(orient='records')}")

    # Preprocess amenities
    places = places.apply(lambda x: x.astype(str).str.lower().str.strip())
    places["amenities"] = places["amenities"].str.replace('.', '')
    places["amenities"] = places["amenities"].str.replace(" ", "")
    places["amenities"] = places["amenities"].apply(lambda x: x.split(","))
    places["amenities"] = places["amenities"].apply(lambda x: [item.strip() for item in x])
    places["amenities"] = places["amenities"].apply(lambda x: sorted(x))
    places["amenities"] = places["amenities"].apply(lambda x: " ".join(x))

    logger.info(f"Preprocessed amenities: {places['amenities'].tolist()}")

    # Fetch user's saved destinations
    user_saved_destinations = list(saved_destinations.find({"user_id": user_id}, {"destination_id": 1}))
    user_saved_destinations = [ObjectId(doc["destination_id"]) for doc in user_saved_destinations]

    logger.info(f"User's saved destination IDs: {user_saved_destinations}")

    # Fetch details of saved destinations
    saved_destinations_details = list(destinations.find({"_id": {"$in": user_saved_destinations}}, {"_id": 1, "category": 1, "amenities": 1}))
    logger.info(f"Details of saved destinations: {saved_destinations_details}")

    # Create user profile
    user_profile = pd.DataFrame({
        "_id": [user_id],
        "savedDestinations": [user_saved_destinations]
    })

    user_profile["preferredAmenities"] = user_profile["savedDestinations"].apply(lambda x: find_amenities(x, places))

    logger.info(f"User's preferred amenities: {user_profile['preferredAmenities'].tolist()}")

    user = user_profile.loc[user_profile["_id"] == user_id]

    if user.empty:
        logger.info("User has no saved destinations. Generating random recommendations.")
        if places.shape[0] >= 10:
            recomend_list = []
            while len(recomend_list) < (5 * days):
                indx = randint(0, places.shape[0] - 1)
                if places["category"].iloc[indx] in category:
                    recomend_list.append(str(places["_id"].iloc[indx]))
            return jsonify(list(set(recomend_list)))
        else:
            return jsonify([str(id) for id in places["_id"].to_list()])

    # Calculate TF-IDF and cosine similarity
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    place_tfidf = tfidf.fit_transform(places["amenities"])
    user_tfidf = tfidf.transform(user_profile["preferredAmenities"])
    cosine_sim = cosine_similarity(user_tfidf, place_tfidf)

    sim_scores = list(enumerate(cosine_sim[-1, :]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    logger.info(f"Cosine similarity scores: {sim_scores}")

    # Generate recommendations
    i = 0
    recomend_list = [str(id) for id in user["savedDestinations"].iloc[0]]
    while len(recomend_list) < (5 * days):
        if i >= len(sim_scores):
            break
        indx = sim_scores[i][0]
        if places["category"].iloc[indx] in category:
            if str(places["_id"].iloc[indx]) not in recomend_list:
                recomend_list.append(str(places["_id"].iloc[indx]))
        i += 1

    # If the list is still short, add random destinations from the selected category
    if len(recomend_list) < (5 * days):
        remaining = (5 * days) - len(recomend_list)
        available_destinations = places[places["category"].isin(category)]
        
        logger.info(f"Available destinations in selected categories: {available_destinations.to_dict(orient='records')}")

        # Check if there are enough destinations to sample
        if len(available_destinations) >= remaining:
            additional_destinations = available_destinations.sample(remaining)
        else:
            # If not enough, add all available destinations
            additional_destinations = available_destinations
        
        recomend_list.extend(additional_destinations["_id"].astype(str).tolist())

    logger.info(f"Final recommendations: {recomend_list}")

    return jsonify(recomend_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)  # Disable debug mode in production