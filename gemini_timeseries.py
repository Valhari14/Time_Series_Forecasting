import google.generativeai as genai
from pathlib import Path
import streamlit as st

headers ={
    "authorization": st.secrets["API_KEY"],
    "content-type": "application/json"
}
genai.configure(api_key=headers["authorization"])

# Set up the model
generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}


model = genai.GenerativeModel(model_name="gemini-pro-vision",
                              generation_config=generation_config,
                              )

image_path = Path("image.jpeg")

prompt = "The image showcases time series forecasting for the number of accidents over time, derived from the accident reports dataset of Karnataka, India. Assume the role of a Data Analyst and provide key observations and insights in english to aid police and traffic department in better decision making to ultimately reduce the accidents.Give detailed point explanation for each of the below mentioned points uniquely for Karnataka.1. Overall Trend Analysis: Identify significant trends in accident rates over the years. Highlight increases or decreases with COVID-19 impact.2. Seasonal Pattern : Analyze how seasonal variations in the seasons in Karnataka, like monsoons, major festivals of Karnataka, impact accident rates.3. Monthly Variations: Investigate recurring patterns in accident rates by month of the year and potential reasons. 4. Daily Accident Trends: Mention the specific days or time frames with elevated accident rates. Differentiate between weekdays and weekends to understand distinct accident patterns and the potential reasons behind them. Analyze the time and days of the week and potential reasons behind the pattern.5. Additional Insights and Recommendations: Give in bullet points in detail.Provide targeted recommendations for resource allocation based on identified trends and peak periods. Suggest traffic police interventions, such as targeted patrols or awareness campaigns during high-risk periods. Propose strategies for junction control, traffic signal optimization, and overall traffic management to reduce accidents. Discuss deployment strategies for emergency services and coordination mechanisms during peak accident times."


image_part = {
    "mime_type" : "image/jpeg",
    "data" : image_path.read_bytes()
}

prompt_parts = [
    prompt , image_part
]

response = model.generate_content(prompt_parts)
print(response.text)