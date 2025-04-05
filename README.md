# it4490-prototype
An AI-powered Career Assistant prototype.

The interface offers three core features:
- **Job Advisor:** A chatbot that suggests potential jobs based on a user's skills and interests.
- **Resume Grammar Checker:** Users upload a PDF file of a resume and the AI will output grammar corrections or wording improvements.
- **Cover Letter Generator:** Users input the job title they're applying for, and the AI will generate a tailored cover letter.

The main logic is implemented in career-app.py, which has an implemented interface (Gradio) that gives users a friendly UI to navigate to the specific features via a tab bar.

career-app.py - central file that hosts Gradio along with the Job Advisor AI
pdfhandler.py - secondary file that handles pdf processing.
