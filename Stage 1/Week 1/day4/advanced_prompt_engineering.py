import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_response(prompt, temp=0.7, top_p=0.9, freq_pen=0.0, pres_pen=0.0):
    """ Sends a prompt to OpenAI API with tuning parameters """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen
    )
    return response["choices"][0]["message"]["content"]

# Example prompts
default_prompt = "Explain the significance of Artificial Intelligence in modern technology."
creative_prompt = "Imagine AI is a living entity. How would it describe itself?"

#Example: Role-Based Prompting
#role_prompt = "You are an experienced AI researcher. Explain Quantum AI to a beginner."
#expert_response = get_gpt_response(role_prompt, temp=0.5)
#print("Expert Role Response:", expert_response)


# Experimenting with different parameters
print("Default Response:", get_gpt_response(default_prompt))
print("Creative High-Temperature Response:", get_gpt_response(creative_prompt, temp=1.0))
print("Strict Logical Response:", get_gpt_response(default_prompt, temp=0.1, top_p=0.5))
print("Diverse Topic Exploration:", get_gpt_response(default_prompt, pres_pen=1.5))
