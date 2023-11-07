import streamlit as st
from transformers import pipeline, set_seed, Conversation
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import string
import nltk
from nltk.corpus import stopwords
from geopy.geocoders import Nominatim
from math import radians, cos, sin, asin, sqrt
from geopy.exc import GeocoderInsufficientPrivileges, GeocoderTimedOut, GeocoderServiceError



if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Food Recommendation Chatbot")

# Load the SentenceTransformer model, business data, and conversation pipeline
model1 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
df_business_ca = pd.read_csv('/Users/mrunmayeerane/Desktop/progress/Flavors/business_in_ca.csv')  # Update the path to your CSV file
conversation_pipeline = pipeline("conversational", model="microsoft/DialoGPT-medium")
set_seed(42)
nltk.download('punkt')
nltk.download('stopwords')

with open('yelp_recommendation_model_8.pkl', 'rb') as file:
    P = pickle.load(file)  # User embeddings or features
    Q = pickle.load(file)  # Business embeddings or features
    embeddings_userid_array = pickle.load(file) 

def clean_text_data(texts):
    stop_words = set(stopwords.words('english'))
    
    for text in texts:
        # Check characters to see if they are in punctuation
        nopunc = [char for char in texts if char not in string.punctuation]
    
        # Join the characters again to form the string.
        nopunc = ''.join(nopunc)
    

    return " ".join([word for word in nopunc.split() if word.lower() not in stop_words])

def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

# Define the function to get recommendations
def get_recommendations(input_text, user_lat, user_lon):
    cleaned_input_text = clean_text_data(input_text)
    input_embeddings = model1.encode([cleaned_input_text])
    # Assume Q is your business feature matrix
    ratings = np.dot(input_embeddings, Q.T)
    predictItemRating = pd.DataFrame(ratings, columns=Q.index)
    topRecommendations = predictItemRating.T.sort_values(by=0, ascending=False).head(7)
    top_recommendations_df = topRecommendations.rename_axis('business_id').reset_index()
    merged_results = pd.merge(
    top_recommendations_df,
    df_business_ca[['business_id', 'name', 'stars', 'categories', 'hours', 'latitude', 'longitude']],
    on='business_id',
    how='left')
    merged_results.columns = ['business_id', 'rating', 'business_name', 'stars', 'categories', 'hours', 'latitude', 'longitude']
    if user_lat is not None and user_lon is not None:
        # Calculate distance and add it to the DataFrame
        merged_results['distance'] = merged_results.apply(
            lambda row: haversine(user_lon, user_lat, row['longitude'], row['latitude']),
            axis=1
        )
        # Sort by distance
        merged_results.sort_values('distance', inplace=True)
    # recommendations_string = ", ".join(
    #     f"{row['business_name']} (Stars: {row['stars']})" for _, row in merged_results.iterrows()
    # )
    # return recommendations_string
    recommendations_string = ", ".join(
        f"{row['business_name']} (Stars: {row['stars']}, Distance: {row['distance']:.2f} km)" 
        for _, row in merged_results.iterrows()
    )
    return recommendations_string

def geocode_address(address):
    try:
        geolocator = Nominatim(user_agent="my_unique_geocode_app")  # Use a unique user-agent
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except (GeocoderInsufficientPrivileges, GeocoderTimedOut, GeocoderServiceError) as e:
        st.error(f"Geocoding error: {e}")
        return None, None


def handle_conversation(user_input, user_lat, user_lon):
    # Initialize response_text with a default value
    response_text = "Hi! Tell me what you're looking for, and I'll recommend the best places for you!"
    
    if user_input.lower().startswith("recommend"):
        # Handle recommendation request
        if user_lat is not None and user_lon is not None:
            recommendations = get_recommendations(user_input, user_lat, user_lon)
        else:
            recommendations = "Sorry, I couldn't determine your location for nearby recommendations."

        # recommendations = get_recommendations(user_input, user_lat, user_lon)
        if recommendations:
            
            # Create a response including the recommendations
            response_text = f"Based on your preference, I recommend {recommendations}. They are all great choices! Is there anything else you need help with?"
           
           
    else:
        # Generate conversational response
        if 'history' not in st.session_state or not st.session_state.history:
            st.session_state.history = [Conversation()]
        
        conversation = st.session_state.history[-1]
        conversation.add_user_input(user_input)
        new_response = conversation_pipeline(conversation)
        
        response_text = new_response.generated_responses[-1]
        st.session_state.history[-1] = conversation
    
    return response_text



prompt = st.chat_input("Hi! Tell me what you're looking for, and I'll recommend the best places for you.")
user_location = st.text_input("Enter your address for nearby recommendations.")
if prompt and user_location:
    user_lat, user_lon = geocode_address(user_location)
    if user_lat is not None and user_lon is not None:
        response = handle_conversation(prompt, user_lat, user_lon)
    else:
        response = "Sorry, we couldn't find your location. Please check the address."

    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Determine the response based on user input
    # response = handle_conversation(prompt)
    # response = handle_conversation(prompt, user_lat, user_lon) 
    
    # Append the response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



