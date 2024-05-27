import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import PyPDF2

# Charger le modèle et le tokenizer GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction pour générer les réponses GPT-2
def generate_response(prompt, max_new_tokens=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if inputs.size(1) > 1024 - max_new_tokens:
        inputs = inputs[:, -1024 + max_new_tokens:]
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Fonction pour extraire le texte des fichiers PDF
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            texts.append(text.strip())
    return texts

# Fonction pour ajouter un message au chat
def add_message(message, sender="User"):
    chat_text.config(state=tk.NORMAL)
    chat_text.insert(tk.END, f"{sender}: {message}\n")
    chat_text.config(state=tk.DISABLED)
    chat_text.yview(tk.END)

# Fonction pour envoyer un message
def send_message(event=None):
    user_input = user_entry.get("1.0", tk.END).strip()
    if user_input == "":
        return
    add_message(user_input, "User")
    user_entry.delete("1.0", tk.END)
    process_user_input(user_input)

# Fonction pour traiter l'entrée utilisateur
def process_user_input(user_input):
    prompt = f"Analyse le texte suivant:\n{user_input}\n\nAnalyse complète :"
    analysis = generate_response(prompt)
    add_message(f"Analyse: {analysis}", "Bot")

# Fonction pour sélectionner les fichiers PDF et extraire leur texte
def select_pdfs():
    pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    if pdf_paths:
        extracted_texts = extract_text_from_pdfs(pdf_paths)
        for text in extracted_texts:
            add_message(f"Texte extrait du PDF:\n{text}", "Bot")
            user_entry.insert(tk.END, text + "\n\n")

# Création de l'interface utilisateur
root = tk.Tk()
root.title("Chatbot d'Analyse de Textes")

chat_frame = tk.Frame(root)
chat_frame.pack(padx=10, pady=5, fill="both", expand=True)

chat_text = scrolledtext.ScrolledText(chat_frame, state=tk.DISABLED, wrap=tk.WORD)
chat_text.pack(padx=5, pady=5, fill="both", expand=True)

user_entry = tk.Text(root, height=5, wrap=tk.WORD)
user_entry.pack(padx=5, pady=5, fill="x")

button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=5, fill="x")

send_button = tk.Button(button_frame, text="Envoyer", command=send_message)
send_button.pack(side="left", padx=5)

pdf_button = tk.Button(button_frame, text="Sélectionner des fichiers PDF", command=select_pdfs)
pdf_button.pack(side="left", padx=5)

root.mainloop()
