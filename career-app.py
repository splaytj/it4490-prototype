import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdfhandler import process_resume
from cvrletgen import generate_cover_letter
import gradio as gr

# Load chatbot model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def create_cover_letter(job_title, company_name, user_name, skills):
    return generate_cover_letter(job_title, company_name, user_name, skills)

# Chatbot setup -------------------------------------------------------------------------------------------------------------------------------------------------------
system_prompt = "You are a job advisor helping users find careers that match their interests and skills. Provide clear, relevant job suggestions."
initial_message = "Hello! I'm a job advisor here to help you find the perfect job for you. Let's start by talking about your interests and skills.\n" \
                  "For example, someone interested in computers and problem-solving might enjoy roles like Software Engineer, Data Analyst, or Cybersecurity Specialist.\n" \
                  "Let's start! What are your interests and strengths?"
chat_history_ids = tokenizer.encode(system_prompt + tokenizer.eos_token + initial_message + tokenizer.eos_token, return_tensors='pt')

def chatbot_response(user_input, chat_history_ids):
    """Generate a chatbot response."""
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
        temperature=0.6,
        repetition_penalty=1.2
    )
    
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response, chat_history_ids

# Gradio interface --------------------------------------------------------------------------------------------------------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown("# AI Job Advisor and Resume Grammar Checker")

    # Resume Grammar Checker tab
    with gr.Tab("Resume Grammar Checker"):
        pdf_upload = gr.File(label="Upload your resume (PDF)")
        output_html = gr.HTML(label="Grammar Corrections")
        check_button = gr.Button("Check Grammar")
        
        check_button.click(process_resume, inputs=pdf_upload, outputs=output_html)

    # Cover Letter Generator tab
    with gr.Tab("Cover Letter Generator"):
        gr.Markdown("### Generate a Professional Cover Letter")
        job_title_input = gr.Textbox(label="Job Title", placeholder="e.g., Software Engineer")
        company_name_input = gr.Textbox(label="Company Name", placeholder="e.g., TechCorp")
        user_name_input = gr.Textbox(label="Your Full Name", placeholder="e.g., Jane Doe")
        skills_input = gr.Textbox(label="Your Key Skills", placeholder="e.g., Python, problem-solving, teamwork")
        submit_button = gr.Button("Generate Cover Letter")
        output = gr.Textbox(label="Your Cover Letter", lines=15, placeholder="Your cover letter will appear here...")

        submit_button.click(
            fn=create_cover_letter,
            inputs=[job_title_input, company_name_input, user_name_input, skills_input],
            outputs=output
        )
    
    # Job Advisor Chatbot tab
    with gr.Tab("Job Advisor Chat") as job_tab:
        chatbot = gr.Chatbot()
        user_input = gr.Textbox(label="Your message")
        chat_state = gr.State(chat_history_ids)

        def show_initial_message():
            return[(None, initial_message)]
        
        def chat(user_input, chat_state):
            response, new_chat_state = chatbot_response(user_input, chat_state)
            return [(user_input, response)], new_chat_state
        
        user_input.submit(chat, inputs=[user_input, chat_state], outputs=[chatbot, chat_state])
    
        job_tab.select(fn=show_initial_message, outputs=chatbot)

app.launch()