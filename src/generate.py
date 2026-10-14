import os
import re
import threading

import documents  # noqa: F401 - first, so the models folder is set before transformers loads
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from retrieve import retrieve_docs

# Disable Hugging Face symlink warning (Windows compatibility fix)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load lightweight chat model (TinyLlama)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Automatically detect GPU (CUDA) or fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with optimized precision based on hardware
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True
)
# Move model to selected device (GPU/CPU)
model = model.to(device)

# One generation at a time - the API serves requests on several threads
_generate_lock = threading.Lock()

# Retrieved context sent to the model. Whole chunks only, so none is cut mid-sentence.
MAX_CONTEXT_CHARS = 1800

# "tutor" = the Streamlit CS tutor: uses the documents when relevant, otherwise
# answers from the model's own knowledge.
# "documents" = DocuChat (rag_api.py): answers ONLY from the documents and says
# so when they don't cover the question, so every answer has a source.
PERSONAS = {
    "tutor": {
        "system": (
            "You are a computer science tutor. Answer the student's question clearly "
            "and briefly. Use the context excerpts when they are relevant."
        ),
        "greeting": "Hello! How can I help you with computer science today?",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "about": "I'm your AI computer science tutor. Ask me anything about programming or CS.",
        "grounded": False,
    },
    "documents": {
        "system": (
            "You are a document assistant. Answer using ONLY the context excerpts from "
            "the user's documents. If the context does not contain the answer, say you "
            "couldn't find it in the documents. Keep the answer short."
        ),
        "greeting": "Hello! Ask me anything about your documents.",
        "thanks": "You're welcome! Let me know if you have more questions.",
        "about": "I'm an assistant that answers questions from your uploaded documents.",
        "not_found": "I couldn't find anything about that in your documents.",
        "grounded": True,
    },
}

# Predefined responses for greetings, thanks, and basic small talk
SMALL_TALK = [
    (("hi", "hello", "hey", "hiya", "greetings", "good morning", "good afternoon", "good evening"), "greeting"),
    (("thanks", "thank you", "thx", "ty", "cheers"), "thanks"),
    (("how are you", "who are you", "what are you"), "about"),
]
# Words that may follow a small-talk phrase without making it a real question
# ("hi there", "thank you so much", "how are you doing today")
FILLER = {
    "there", "again", "so", "much", "a", "lot", "very", "for", "the", "your", "help",
    "you", "all", "everyone", "guys", "bot", "friend", "buddy", "doing", "today", "anyway",
}

# Handles basic conversational inputs without using the model. Matches whole
# words and only when the message is nothing BUT small talk - so "hi" inside
# "machine" or "this", or "hi, what is a stack?", still gets a real answer.
def handle_small_talk(query, mode="tutor"):
    text = " ".join(re.findall(r"[a-z']+", query.lower()))

    for phrases, kind in SMALL_TALK:
        for phrase in phrases:
            if text == phrase or text.startswith(phrase + " "):
                rest = text[len(phrase):].split()
                if all(word in FILLER for word in rest):
                    return PERSONAS[mode][kind]

    return None

# Joins retrieved chunks into the context block, stopping before the limit
# instead of cutting a chunk in half. Returns the context and the chunks used.
def build_context(retrieved, max_chars=MAX_CONTEXT_CHARS):
    parts, used, total = [], [], 0
    for r in retrieved:
        part = f"[Source {len(parts) + 1}: {r['filename']}]\n{r['content']}"
        if parts and total + len(part) > max_chars:
            break
        parts.append(part)
        used.append(r)
        total += len(part) + 2
    return "\n\n".join(parts), used

# Runs the chat model on one system + user message and returns only the new text
def run_model(system, user):
    # TinyLlama-Chat was trained on this chat format; a plain-text prompt makes it
    # continue the text instead of answering
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize input and move to correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response using the language model (greedy decoding: same question, same answer)
    with _generate_lock:
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the tokens generated after the prompt. Slicing the decoded text
    # by len(prompt) breaks, because decoding doesn't reproduce the prompt exactly.
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# Retrieves relevant documents and generates an answer.
# Returns (answer, chunks the answer was based on).
def answer_question(query, k=5, mode="tutor"):
    persona = PERSONAS[mode]

    # handle conversation first
    small_talk_response = handle_small_talk(query, mode)
    if small_talk_response:
        return small_talk_response, []

    context, used = build_context(retrieve_docs(query, k=k))

    if not used:
        if persona["grounded"]:
            return persona["not_found"], []
        user = f"Question:\n{query}"
    else:
        user = f"Context:\n{context}\n\nQuestion:\n{query}"

    return run_model(persona["system"], user), used

# Answer text only (used by the Streamlit app and the terminal loop below)
def generate_answer(query, k=5):
    answer, _ = answer_question(query, k=k)
    return answer

# Simple terminal interface for continuous user interaction
if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        print(generate_answer(q))
