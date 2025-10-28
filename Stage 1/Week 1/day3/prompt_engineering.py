import openai
import os
from dotenv import load_dotenv

# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_gpt_response(prompt, model="gpt-3.5-turbo"):
    """ Sends a prompt to OpenAI API and returns response """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

# Define prompts
zero_shot_prompt = "Translate 'Goodbye' to French."
few_shot_prompt = "English: Hello → French: Bonjour\nEnglish: Thank you → French: Merci\nEnglish: Goodbye → French: "
chain_of_thought_prompt = "Q: If I have 10 apples and I eat 3, how many do I have left? Think step by step before answering."

# Run API requests
print("Zero-Shot Response:", get_gpt_response(zero_shot_prompt))
print("Few-Shot Response:", get_gpt_response(few_shot_prompt))
print("Chain-of-Thought Response:", get_gpt_response(chain_of_thought_prompt))
