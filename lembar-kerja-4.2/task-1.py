from openai import OpenAI

client = OpenAI(api_key="sk-...")  # Ganti dengan API key milikmu

response = client.chat.completions.create(
    model="gpt-3.5-turbo",  # atau gpt-4 jika punya akses
    messages=[
        {"role": "system", "content": "Kamu adalah asisten yang membantu."},
        {"role": "user", "content": "Apa itu machine learning?"}
    ]
)

print(response.choices[0].message.content)
