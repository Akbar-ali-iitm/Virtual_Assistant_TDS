import os
import base64
import time
import numpy as np
from fastapi import FastAPI, Request
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

load_dotenv()

app = FastAPI()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://aipipe.org/openai/v1")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-0125"

# --- Rate Limiter ---
class RateLimiter:
    def __init__(self, rpm=60, rps=2):
        self.rpm = rpm
        self.rps = rps
        self.request_times = []
        self.last = 0

    def wait(self):
        now = time.time()
        if now - self.last < 1 / self.rps:
            time.sleep((1 / self.rps) - (now - self.last))
        self.request_times = [t for t in self.request_times if now - t < 60]
        if len(self.request_times) >= self.rpm:
            time.sleep(60 - (now - self.request_times[0]))
        self.request_times.append(now)
        self.last = time.time()

rate_limiter = RateLimiter(rpm=5, rps=2)

# --- Load Embeddings ---
def load_embeddings():
    data = np.load("embeddings.npz", allow_pickle=True)
    return data["chunks"], np.vstack(data["embeddings"]), data["metadata"]

# --- Embed Text ---
def get_embedding(text, retries=3):
    enc = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    for attempt in range(retries):
        try:
            rate_limiter.wait()
            if len(enc.encode(text)) > 8192:
                raise ValueError("Input too long")
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
                dimensions=512
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

# --- Image Captioning ---
def get_image_caption(base64_img):
    messages = [
        {"role": "system", "content": "You are an OCR and captioning assistant. Describe the contents of the image including visible text, UI elements, and instructions."},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# --- LLM Response ---
def generate_llm_response(question, context):
    system_prompt = (
        "You are a knowledgeable and concise teaching assistant. Use ONLY the information in the context to answer.\n"
        "* Respond in **Markdown**.\n"
        "* Use bullet points or numbered lists if helpful.\n"
        "* Wrap code in triple backticks.\n"
        "\n"
        "⚠️ If you cannot answer, say exactly:\n"
        "```\n**I'm not sure based on the course material provided.** Try rephrasing your question or check the [TDS Discourse forum](https://discourse.onlinedegree.iitm.ac.in/c/courses/tds-kb/34).\n```"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    chat = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=512
    )
    return chat.choices[0].message.content.strip()

# --- Main Answer Function ---
def answer(question, image=None):
    chunks, embeddings, metas = load_embeddings()

    # Append image caption if provided
    if image:
        try:
            caption = get_image_caption(image)
            question += f" {caption}"
        except Exception as e:
            print(f"Image captioning failed: {e}")

    # Get question embedding
    q_embed = get_embedding(question)

    # Find top similar chunks
    sims = np.dot(embeddings, q_embed) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embed)
    )
    top_idxs = np.argsort(sims)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_idxs]
    top_links = [
        {
            "url": metas[i].get("url", ""),
            "text": metas[i].get("text", "")[:100]
        }
        for i in top_idxs if "url" in metas[i]
    ]

    # Generate answer
    answer_text = generate_llm_response(question, "\n\n".join(top_chunks))

    return {
        "answer": answer_text,
        "links": top_links
    }

# --- FastAPI Endpoint ---
@app.post("/api/")
async def api_handler(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "")
        image = data.get("image", None)
        return answer(question, image)
    except Exception as e:
        return {
            "answer": (
                "**Something went wrong while processing your question.** "
                f"Please try again. Error: `{str(e)}`"
            ),
            "links": []
        }
