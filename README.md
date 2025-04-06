# it4490-prototype (CareerMate)
An AI-powered Career Assistant prototype called 'CareerMate'.

The interface offers three core features:
- **Job Advisor:** A chatbot that suggests potential jobs based on a user's skills and interests.
- **Resume Grammar Checker:** Users upload a PDF file of a resume and the AI will output grammar corrections or wording improvements.
- **Cover Letter Generator:** Users input information for the job they are applying for, and the AI will generate a tailored cover letter.

app.py - central file that hosts Gradio and connects all other files.

pdfhandler.py - secondary file that handles pdf processing using the Prithivi Da GEC model.

cvrletgen.py - secondary file that handles cover letter generation using the GPT-2 model.

jobadvisor.py - seocndary file that handles the chatbot using the GPT-2 model.


Also available on Hugging Face Spaces: https://huggingface.co/spaces/splaytj/IT-4490-Prototype
**DISCLAIMER**: Please allow up to 10 minutes for the application to fully load if not yet built. Additionally, we've attached a sample resume to upload to the Grammar Checker.
