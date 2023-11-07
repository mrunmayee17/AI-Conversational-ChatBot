# AI-Conversational-ChatBot
# Food Recommendation Chatbot
<img width="1431" alt="Screenshot 2023-11-07 at 3 08 20 AM" src="https://github.com/mrunmayee17/AI-Conversational-ChatBot/assets/48186569/d534e3d2-5842-48af-a180-cde9547ac069">

## Overview
This Food Recommendation Chatbot is designed to provide restaurant suggestions based on user preferences and location. It utilizes natural language processing and geocoding to offer a seamless and interactive experience to users.

## Features
- Conversational Interaction: Engage with users in a natural, conversational manner.
- Location-Based Recommendations: Provide restaurant suggestions near the user's current location.
- NLP for User Input Processing: Analyze and interpret user requests using advanced NLP techniques.

## Dependencies
To run this chatbot, the following Python libraries are required:
- streamlit
- transformers
- sentence_transformers
- pandas
- numpy
- pickle
- nltk
- geopy

## Installation
1. Install Python 3.x from the official website.
2. Install the necessary packages using pip:
   ```bash
   pip install streamlit transformers sentence_transformers pandas numpy pickle nltk geopy

## Download NLTK Data
To use the chatbot, you need to download necessary NLTK data. Run the following Python commands:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords') 

## Usage
To run the Streamlit app, execute the following command in your terminal:
```bash
streamlit run your_script_name.py 

## Required Files
1. business_in_ca.csv: Contains business data for California.
2. yelp_recommendation_model_8.pkl: A pickle file with user and business embeddings.

## Key Functions
1. clean_text_data(texts): Cleans and processes text data.
2. haversine(lon1, lat1, lon2, lat2): Calculates distances between two geographical points.
3. get_recommendations(input_text, user_lat, user_lon): Provides restaurant recommendations.
4. geocode_address(address): Converts an address into geographical coordinates.
5. handle_conversation(user_input, user_lat, user_lon): Manages conversation logic and user interactions.

## Streamlit Components
1. Chat Input: Where users can type their queries or requests.
2. Text Input: For users to enter their address for location-specific recommendations.
3. Chat History: Displays the ongoing conversation.

## Limitations
1. The chatbot's recommendations are currently limited to California.
2. It relies on accurate geocoding for location-based suggestions.


## Future Improvements
1. Expanding the geographical scope of the chatbot.
2. Implementing more sophisticated NLP models for enhanced user query understanding.
3. For further assistance or queries, feel free to reach out to the support team or refer to the project documentation.
