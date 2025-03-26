import os
import streamlit as st
import requests
from dotenv import load_dotenv
from langchain.llms import BaseLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import ClassVar, List, Optional
from langchain.schema import Generation, LLMResult

# Load environment variables from .env file
load_dotenv()

# ------------------------------
# CONFIGURATION: Read API keys from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")

# ------------------------------
# Custom Gemini LLM Wrapper using LangChain
class GeminiLLM(BaseLLM):
    api_url: ClassVar[str] = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    api_key: str

    def _call(self, prompt, stop: Optional[List[str]] = None) -> str:
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        cleaned_prompt = prompt.strip()
        request_data = {
            "contents": [
                {"role": "user", "parts": [{"text": cleaned_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 500,
                "topP": 0.95,
                "topK": 40
            }
        }
        full_url = f"{self.api_url}?key={self.api_key}"
        response = requests.post(full_url, json=request_data)
        # Debug output removed for final UI
        json_resp = response.json()
        if "candidates" in json_resp and len(json_resp["candidates"]) > 0:
            candidate = json_resp["candidates"][0]
            if candidate.get("content") and candidate["content"].get("parts"):
                parts = candidate["content"]["parts"]
                if isinstance(parts, list):
                    return parts[0].get("text", "")
                elif isinstance(parts, dict):
                    return parts.get("text", "")
        return "No response from Gemini API."

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _generate(self, prompt: str, stop: Optional[List[str]] = None) -> LLMResult:
        text = self._call(prompt, stop=stop)
        generation = Generation(text=text)
        return LLMResult(generations=[[generation]])

# Instantiate the Gemini LLM
llm = GeminiLLM(api_key=GEMINI_API_KEY)

# ------------------------------
# Google API Helper Functions
def get_places_suggestions(destination, query="tourist attraction"):
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={destination}&key={GOOGLE_PLACES_API_KEY}"
    geocode_response = requests.get(geocode_url).json()
    if geocode_response.get("results"):
        location = geocode_response["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]
    else:
        st.error("Geocoding failed for the destination.")
        return []
    
    full_query = query if query.strip() != "" else "tourist attraction"
    places_url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
        f"location={lat},{lng}&radius=10000&type=tourist_attraction&keyword={full_query}&key={GOOGLE_PLACES_API_KEY}"
    )
    places_response = requests.get(places_url).json()
    suggestions = []
    if places_response.get("results"):
        for place in places_response["results"]:
            suggestions.append({
                "name": place.get("name"),
                "address": place.get("vicinity"),
                "rating": place.get("rating", "N/A")
            })
    return suggestions

# ------------------------------
# Updated Itinerary Prompt Template
# This prompt instructs the model to output an itinerary in a fixed format.
itinerary_prompt_template = PromptTemplate(
    input_variables=["budget", "trip_duration", "start_location", "destination", "purpose", "preferences"],
    template="""\
Okay, here's a concise {trip_duration}-day {destination} itinerary tailored to your preferences and {budget} budget, focusing on {purpose} starting from {start_location}:

**Day 1:** [Provide one brief sentence covering essential activities, dining recommendations, and transportation tips based on preferences: {preferences}]

**Day 2:** [Provide one brief sentence summary]

...
**Day {trip_duration}:** [Provide one brief sentence summary]

Keep the response short, focused, and clear.
"""
)
itinerary_chain = LLMChain(llm=llm, prompt=itinerary_prompt_template)

# ------------------------------
# Refined Suggestions Prompt Template
refined_suggestions_template = PromptTemplate(
    input_variables=["preferences", "places_list", "budget"],
    template="""\
Based on the user's preferences ({preferences}) and budget ({budget}), and the following list of attractions:
{places_list}
Provide a concise markdown bullet list with at least 2-3 top attractions that best match the user's desired travel style.
Include each attraction's name, its rating, and a one-line note.
"""
)
refined_suggestions_chain = LLMChain(llm=llm, prompt=refined_suggestions_template)

# ------------------------------
# Streamlit Application UI
st.title("Agentic AI Travel Itinerary Planner with Gemini & LangChain")

st.header("Enter Your Travel Details")
budget = st.selectbox("Budget", ["Low", "Moderate", "High"])
trip_duration = st.number_input("Trip Duration (in days)", min_value=1, max_value=30, value=4)
start_location = st.text_input("Starting Location", "Hyderabad")
destination = st.text_input("Destination", "Kerala")
purpose = st.text_area("Purpose of Travel", "Cultural exploration, historical sites, and local food.")
preferences = st.text_area("Additional Preferences", "Hidden gems, food recommendations, budget-friendly public transport")

# ------------------------------
# Get Travel Suggestions Section
if st.button("Get Travel Suggestions"):
    st.info("Analyzing your preferences and matching desired results...")
    raw_places = get_places_suggestions(destination, query=f"tourist attraction {preferences}")
    if raw_places:
        places_list = ""
        for place in raw_places[:5]:
            places_list += f"- {place['name']} (Address: {place['address']}, Rating: {place['rating']})\n"
        refined_suggestions = refined_suggestions_chain.run({
            "preferences": preferences,
            "places_list": places_list,
            "budget": budget
        })
        st.subheader("Your Refined Travel Suggestions")
        st.markdown(refined_suggestions)
    else:
        st.error("No suggestions available from Google Places API.")

# ------------------------------
# Generate Detailed Itinerary Section
if st.button("Generate Detailed Itinerary"):
    st.info("Generating a personalized itinerary based on your preferences...")
    with st.spinner("Creating your itinerary..."):
        itinerary_outline = itinerary_chain.run({
            "budget": budget,
            "trip_duration": trip_duration,
            "start_location": start_location,
            "destination": destination,
            "purpose": purpose,
            "preferences": preferences
        })
    st.subheader("Your Detailed Itinerary Outline")
    st.markdown(itinerary_outline)

st.markdown("---")
st.caption("This prototype integrates LangChain with a custom Gemini LLM and Google Places API. Ensure your API keys are valid and authorized.")
