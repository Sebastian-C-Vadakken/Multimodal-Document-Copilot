import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found in .env")

# Configure Gemini
genai.configure(api_key=api_key)

# Pick a model (fast + cheap)
model = genai.GenerativeModel("gemini-2.0-flash")

response = model.generate_content("Reply with a single short greeting line.")

print(response.text)
