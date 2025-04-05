import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# System prompt and initial message
system_prompt = "You are a job advisor helping users find careers that match their interests and skills. Provide clear, relevant job suggestions."
initial_message = "Hello! I'm a job advisor here to help you find the perfect job for you. Let's start by talking about your interests and skills."
print("Bot:", initial_message)

# Initialize conversation history with system prompt and initial message
chat_history_ids = tokenizer.encode(system_prompt + tokenizer.eos_token + initial_message + tokenizer.eos_token, return_tensors='pt')

# Conversation loop
while True:
    # Get user input
    user_input = input("User: ")
    
    # Exit condition
    if user_input.lower() == "quit":
        print("Bot: Goodbye.")
        break
    
    # Encode user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    
    # Limit history to 512 tokens
    if bot_input_ids.shape[-1] > 512:
        bot_input_ids = bot_input_ids[:, -512:]
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.6,
        repetition_penalty=1.2
    )
    
    # Decode and refine response
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    if "job" not in response.lower() and "career" not in response.lower():
        response += " How about a job in that field?"
    print("Bot:", response)