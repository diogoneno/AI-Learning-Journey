import openai
import os

# dont forget when deploying to set environment variable on the hosting platform or use secrets from any available CI/CD workflow (e.g GitHub secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple API call to OpenAI
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="What are the benefits of using LLM APIs?",
    max_tokens=50
)

print(response.choices[0].text.strip())

