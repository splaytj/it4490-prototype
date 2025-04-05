from transformers import pipeline

# Inform the user that the model is being loaded
print("Loading the model, please wait...")

# Load the text generation pipeline with DialoGPT
generator = pipeline('text-generation', model='microsoft/DialoGPT-small')

# Confirm that the model has been loaded
print("Model loaded successfully.")

# Get the job title from the user
job_title = input("Enter the job title: ")

# Inform the user that the cover letter is being generated
print(f"Generating cover letter for {job_title}...")

# Create the prompt for the cover letter
prompt = f"I am writing to apply for the position of {job_title}."

# Generate the cover letter using the model
output = generator(
    prompt,
    max_length=200,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2
)

# Extract the generated text
cover_letter = output[0]['generated_text']

# Print the generated cover letter
print("\nGenerated Cover Letter:\n")
print(cover_letter)