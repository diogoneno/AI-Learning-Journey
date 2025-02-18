from transformers import pipeline

# Load pre-trained model for text generation
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "Artificial intelligence is"
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the output
print(result[0]["generated_text"])
