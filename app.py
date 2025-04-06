from pdfhandler import process_resume
from cvrletgen import generate_cover_letter
from jobadvisor import chat_history_ids, initial_message, chatbot_response
import gradio as gr

def create_cover_letter(job_title, company_name, user_name, skills):
    return generate_cover_letter(job_title, company_name, user_name, skills)

# Gradio interface ------------------------------------------------------------------------------------------------------------
with gr.Blocks() as app:
    gr.Markdown("# CareerMate")

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
