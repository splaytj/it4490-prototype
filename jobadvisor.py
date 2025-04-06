import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load chatbot model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Chatbot setup -------------------------------------------------------------------------------------------------------------------------------------------------------
system_prompt = "You are a job advisor helping users find careers that match their interests and skills. Provide clear, relevant job suggestions."
initial_message = "Hello! I'm a job advisor here to help you find the perfect job for you. Let's start by talking about your interests and skills.\n" \
                  "For example, someone interested in computers and problem-solving might enjoy roles like Software Engineer, Data Analyst, or Cybersecurity Specialist.\n" \
                  "Let's start! What are your interests and strengths?"
chat_history_ids = tokenizer.encode(system_prompt + tokenizer.eos_token + initial_message + tokenizer.eos_token, return_tensors='pt')

# Generate chatbot response
def chatbot_response(user_input, chat_history_ids):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    
    # Limit history to 512 tokens
    if bot_input_ids.shape[-1] > 512:
        bot_input_ids = bot_input_ids[:, -512:]
    
    # Text generation
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=bot_input_ids.shape[-1] + 50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.3,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids
