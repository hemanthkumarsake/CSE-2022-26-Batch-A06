import google.generativeai as genai
import os

# ✅ Use environment variable (SAFE)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
def finance_chatbot(user_message):
    try:
        # ✅ Stable model
        model = genai.GenerativeModel("gemini-pro")

        # ✅ Better prompt
        prompt = f"""
You are FinAI, a smart financial assistant.

Help users with:
- budgeting
- saving money
- expense tracking
- investment advice

Keep answers simple, practical, and friendly.

User Question: {user_message}
"""

        # ✅ Generate response
        response = model.generate_content(prompt)

        # ✅ Handle empty response safely
        if response and hasattr(response, "text"):
            return response.text.strip()
        else:
            return "I couldn't understand. Please try again."

    except Exception as e:
        print("GEMINI ERROR:", str(e))

        # ✅ Handle rate limit
        if "429" in str(e):
            return "Too many requests. Please wait a few seconds."

        # ✅ Handle model errors
        if "404" in str(e):
            return "Model not available. Try again later."

        return "AI is currently unavailable. Please try again later."