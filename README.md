# Humor Generator — RAG + Hugging Face + Telegram Bot  

---

## Overview  

**Humor Generator** is a **Retrieval-Augmented Generation (RAG)** based system that creates, rates, and delivers one-liner jokes — right inside **Telegram**.  
It combines a large language model from **Hugging Face** with a custom humor knowledge base of 25k jokes, letting the AI learn your humor style and score itself like a stand-up critic.  

---

## 🧠 Project Highlights  

| Feature | Description |
|----------|-------------|
| 🧩 **RAG Pipeline** | Retrieves semantically similar jokes from a 25k-entry joke dataset to guide the model. |
| 🤖 **Pre-trained Models** | Uses `mistralai/Mistral-7B-Instruct` for joke generation and `HuggingFaceH4/zephyr-7b-beta` for structured evaluation. |
| 💬 **Telegram Bot Integration** | Chat directly with your AI comedian in Telegram. |
| 📊 **Self-Evaluation** | Model rates every joke on _Funniness_, _Coherence_, and _Originality_. |
| ⚡ **Lightweight** | CPU-friendly; no fine-tuning or GPU dependencies. |
| 🧱 **Modular Design** | Cleanly separated RAG, generation, and evaluation logic. |

---

## 🗂️ Directory Structure  

```

humor-generator/
├── bot.py                # Main Telegram bot script
├── shortjokes1.csv       # Joke dataset (ID, Joke)
├── .env                  # Environment variables (Hugging Face + Telegram tokens)
├── requirements.txt      # Optional: list of dependencies
├── README.md             # You’re reading this!
└── /**pycache**/         # Auto-generated cache (ignored)

````

---

## 🔁 Pipeline Overview  

```mermaid
flowchart TD
    A[User sends topic] --> B[SentenceTransformer embeds topic]
    B --> C[FAISS retrieves top 3 similar jokes]
    C --> D[Mistral-7B-Instruct generates new joke]
    D --> E[Zephyr-7B evaluates joke]
    E --> F[Telegram bot returns joke + ratings]
````

---

## 🧰 Tech Stack

| Layer              | Technology                                                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Backend**        | Python 3.12                                                                                                                                                  |
| **AI Models**      | [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), [Zephyr-7B-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) |
| **Vector DB**      | [FAISS](https://faiss.ai)                                                                                                                                    |
| **Embeddings**     | [Sentence Transformers](https://www.sbert.net) (`all-MiniLM-L6-v2`)                                                                                          |
| **Chat Framework** | [LangChain Hugging Face](https://python.langchain.com)                                                                                                       |
| **Bot API**        | [python-telegram-bot](https://python-telegram-bot.org)                                                                                                       |
| **Env Management** | [python-dotenv](https://pypi.org/project/python-dotenv/)                                                                                                     |

---

## 🚀 Setup & Run

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/magarmach-hash/humor-generator.git
cd humor-generator
```

### 2️⃣ Install Dependencies

Using **uv** or **pip**:

```bash
uv pip install python-telegram-bot==20.7 sentence-transformers faiss-cpu langchain-huggingface python-dotenv pandas numpy
```

### 3️⃣ Add Tokens in `.env`

```bash
HUGGINGFACEHUB_API_TOKEN=hf_your_huggingface_token
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
```

> Get your Telegram token from [@BotFather](https://t.me/BotFather)

### 4️⃣ Run the Bot

```bash
python bot.py
```

Then open Telegram and send your bot a message like:

> programmers and coffee ☕
> aliens using iPhones 👽📱
> exams and depression 📚💀

---

## 🧩 Example Output

```
🤣 Joke about programmers and coffee:
Why do programmers hate coffee breaks?
They can’t handle Java outside the code.

📊 Ratings
Funniness: 4
Coherence: 5
Originality: 4
```

---

## 🧱 Model Pipeline Summary

| Step  | Component           | Description                       |
| ----- | ------------------- | --------------------------------- |
| **1** | SentenceTransformer | Converts topic → embedding vector |
| **2** | FAISS Index         | Retrieves top-k similar jokes     |
| **3** | Mistral-7B          | Generates a new coherent joke     |
| **4** | Zephyr-7B           | Evaluates joke on 3 humor metrics |
| **5** | Telegram Bot        | Delivers response to user         |

---

## 📈 Performance Notes

* Works fully on **CPU** (no CUDA / NVIDIA dependencies)
* Builds FAISS index for 25k jokes in ~30 seconds
* Average response time per joke: ~6–8 seconds
* Handles conversational input like
  *"make a sarcastic joke about exams and sleep"*

---

