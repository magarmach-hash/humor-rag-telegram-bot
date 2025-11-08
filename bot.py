import os
import json
import re
import faiss
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Loading environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Load dataset
df = pd.read_csv("shortjokes1.csv")
docs = df["Joke"].astype(str).tolist()

#RAG setup
embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(np.array(emb, dtype=np.float32))

#joke generation 
llm_gen = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 100
    }
)
chat_gen = ChatHuggingFace(llm=llm_gen)

#LLM for rating 
llm_eval = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.0,
        "max_new_tokens": 60
    }
)
chat_eval = ChatHuggingFace(llm=llm_eval)

def generate_and_rate(topic: str):
    """Generates a joke based on the topic, using RAG context, and then rates it."""
    
    # Retriever
    qv = embedder.encode([topic], convert_to_numpy=True).astype(np.float32)
    _, idxs = index.search(qv, 3)
    ctx = "\n".join(f"- {docs[i]}" for i in idxs[0])

    #joke prompt
    gen_prompt = f"""                                                  
You are a witty comedian.
Here are some example jokes:
{ctx}

Write a new one-liner about "{topic}".
It should be funny, coherent, and original.
Output only the joke.
"""
    joke = chat_gen.invoke(gen_prompt).content.strip()

    # Evaluation prompt
    eval_prompt = f"""
You are a JSON-only rater. 
Rate this joke from 1-5 for Funniness, Coherence, and Originality.

Reply ONLY with valid JSON, like:
{{"Funniness":4,"Coherence":5,"Originality":3}}

Joke: "{joke}"
Topic: "{topic}"
"""
    raw = chat_eval.invoke(eval_prompt).content.strip()
    raw = re.sub(r'```(json)?', '', raw).strip()

    # Parse rating or use defaults
    try:
        ratings = json.loads(raw)
    except Exception:
        # Fallback parsing for non-JSON output
        m = re.findall(r'(\d)', raw)
        ratings = dict(zip(["Funniness", "Coherence", "Originality"], m[:3]))
        if len(ratings) < 3:
            ratings = {"Funniness": "?", "Coherence": "?", "Originality": "?"}

    # Format output message
    return (
        f"*Joke about {topic}:*\n"
        f"{joke}\n\n"
        f"*Ratings*\n"
        f"Funniness: {ratings['Funniness']}\n"
        f"Coherence: {ratings['Coherence']}\n"
        f"Originality: {ratings['Originality']}"
    )

# --- Telegram ---

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    topic = update.message.text.strip()

    if topic.lower() in ["/start", "start"]:
        await update.message.reply_text("Hey! Send me any topic and I'll crack a joke.")
        return

    await update.message.reply_text("Thinking of a good one...")
    result = generate_and_rate(topic)
    await update.message.reply_text(result, parse_mode="Markdown")


if __name__ == '__main__':
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot running...")
    app.run_polling()
