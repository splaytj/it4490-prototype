from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" for better results if your machine can handle it
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_cover_letter(job_title, company_name, user_name, skills):
    # Create a prompt based on user input
    prompt = (
        f"Dear Hiring Manager,\n\n"
        f"I am writing to express my interest in the {job_title} position at {company_name}. "
        f"My name is {user_name}, and I am excited about the opportunity to contribute to your team. "
        f"I bring the following skills to the role: {skills}. Below is a professional cover letter tailored to this position:\n\n"
    )

    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate text using GPT-2
    outputs = model.generate(
        inputs,
        max_length=300,  # Adjust length as needed (cover letters are typically ~250-400 words)
        num_return_sequences=1,
        no_repeat_ngram_size=2,  # Avoid repetition
        do_sample=True,  # Enable sampling for more creative output
        top_k=50,  # Limit to top 50 probable tokens
        top_p=0.95,  # Use nucleus sampling
        temperature=0.7,  # Control randomness (lower = more coherent)
        pad_token_id=tokenizer.eos_token_id  # Handle padding
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process to ensure it ends cleanly
    if not generated_text.endswith("."):
        last_period = generated_text.rfind(".")
        if last_period != -1:
            generated_text = generated_text[:last_period + 1]

    # Add a closing if not present
    if "Sincerely," not in generated_text:
        generated_text += (
            "\n\nThank you for considering my application. I look forward to the opportunity to discuss "
            "how my skills and experience align with the needs of {company_name}. "
            "Sincerely,\n{user_name}"
        ).format(company_name=company_name, user_name=user_name)

    return generated_text

def main():
    # Get user input
    print("Welcome to the AI Cover Letter Generator!")
    job_title = input("Enter the job title: ")
    company_name = input("Enter the company name: ")
    user_name = input("Enter your full name: ")
    skills = input("Enter your key skills (e.g., Python, project management, communication): ")

    # Generate the cover letter
    cover_letter = generate_cover_letter(job_title, company_name, user_name, skills)

    # Print the result
    print("\nHere’s your generated cover letter:\n")
    print(cover_letter)

if __name__ == "__main__":
    main()