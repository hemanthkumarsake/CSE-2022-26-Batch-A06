import google.generativeai as genai
import os

# 1. Configuration - Replace 'YOUR_API_KEY' with your actual key
API_KEY = "AIzaSyCQP0GGWH_LVgFmHREJbVm1bwLlkxksXlU"  # Replace with your real API key
genai.configure(api_key=API_KEY)

def finance_chatbot(user_message):
    try:
        # Initialize the model (Gemini 1.5 Flash for speed/cost)
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Construct a prompt based on the user's message
        prompt = f"""You are a helpful financial assistant. Answer the following finance-related question in a clear, friendly, and helpful manner. Keep your responses concise but informative.

User Question: {user_message}

Financial Advice:"""

        # Generate a response from the model
        response = model.generate_content(prompt)
        
        # Return the generated response text
        return response.text

    except Exception as e:
        # Catch errors related to the API or connection
        print(f"Error: {e}")
        return "Sorry, I couldn't process your request. Please try again later."

# if __name__ == "__main__":
#     # Example question
#     user_question = "How can I save money effectively?"
#     print(finance_chatbot(user_question))