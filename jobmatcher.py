import openai
import os

openai.api_key = "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"

def get_job_recommendation(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a job advisor chatbot."},
                  {"role": "user", "content": user_input}]
    )
    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    while True:
        user_input = input("What are your skills and interests? ")
        if user_input.lower() == "exit":
            break
        print(get_job_recommendation(user_input))
