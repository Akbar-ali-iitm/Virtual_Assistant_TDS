# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi",
#     "httpx",
#     "numpy",
#     "semantic_text_splitter",
#     "uvicorn",
#     "openai",
#     "tiktoken",
#     "pillow",
# ]
# ///

import os
import base64
import time
import numpy as np
from fastapi import FastAPI, Request
from openai import OpenAI
import tiktoken

app = FastAPI()

# --- Setup ---
OPENAI_API_KEY = os.getenv("API_KEY")
MODEL = "text-embedding-3-small"
client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://aipipe.org/openai/v1")

# --- Rate limiter ---
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

# --- Load embeddings ---
def load_embeddings():
    data = np.load("embeddings.npz", allow_pickle=True)
    return data["chunks"], np.vstack(data["embeddings"]), data["metadata"]

# --- Embedding ---
def get_embedding(text, retries=3):
    enc = tiktoken.encoding_for_model(MODEL)
    for attempt in range(retries):
        try:
            rate_limiter.wait()
            tokens = len(enc.encode(text))
            if tokens > 8192:
                raise ValueError("Input too long")
            response = client.embeddings.create(
                model=MODEL,
                input=text,
                dimensions=512
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"⚠️ Retry {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)

# --- Caption Image ---
def get_image_caption(base64_img):
    image_prompt = (
        "You are an OCR and captioning assistant. Describe the contents of the image, including visible text, UI elements, and any instructions."
    )
    messages = [
        {"role": "system", "content": image_prompt},
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
        "You are a knowledgeable and concise teaching assistant. Use only the information provided in the context to answer the question.\n"
        "* Format your response using **Markdown**.\n"
        "* Use code blocks (` ``` `) for any code or command-line instructions.\n"
        "* Use bullet points or numbered lists for clarity where appropriate.\n"
        "* Always include a brief introduction or heading if needed.\n"
        "\n"
        "⚠️ **Important:** If the context does not contain enough information to answer the question, reply exactly with:\n"
        "```\nI don't know\n```"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
    ]
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        temperature=0.5,
        max_tokens=512
    )
    return chat.choices[0].message.content.strip()

# --- Answer Function ---
def answer(question, image=None):
    chunks, embeddings, metas = load_embeddings()
    if image:
        caption = get_image_caption(image)
        question += f" {caption}"

    q_embed = get_embedding(question)
    sims = np.dot(embeddings, q_embed) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_embed)
    )
    top_idxs = np.argsort(sims)[-10:][::-1]
    top_chunks = [chunks[i] for i in top_idxs]
    top_links = [
        {"url": metas[i]["url"], "text": metas[i]["text"][:100]} for i in top_idxs if "url" in metas[i]
    ]
    resp = generate_llm_response(question, "\n\n".join(top_chunks))
    return {
        "answer": resp,
        "links": top_links
    }

# --- API Endpoint ---
@app.post("/api/")
async def api_answer(request: Request):
    try:
        data = await request.json()
        return answer(data.get("question"), data.get("image"))
    except Exception as e:
        return {"error": str(e)}
