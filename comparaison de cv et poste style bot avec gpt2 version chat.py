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
    user_input = user_entry.get()
    if user_input.strip() == "":
        return
    add_message(user_input, "User")
    user_entry.delete(0, tk.END)
    process_user_input(user_input)

# Fonction pour traiter l'entrée utilisateur
def process_user_input(user_input):
    global cv_text, job_text
    if "cv:" in user_input.lower():
        cv_text = user_input[len("cv:"):].strip()
        add_message("CV enregistré. Vous pouvez maintenant entrer une offre d'emploi en commençant par 'offre:'.", "Bot")
    elif "offre:" in user_input.lower():
        job_text = user_input[len("offre:"):].strip()
        add_message("Offre d'emploi enregistrée. Analyse en cours...", "Bot")
        analyze_and_compare()
    else:
        add_message("Commande non reconnue. Veuillez entrer un CV en commençant par 'cv:' ou une offre d'emploi en commençant par 'offre:'.", "Bot")

# Fonction pour analyser et comparer le CV et l'offre d'emploi
def analyze_and_compare():
    try:
        prompt = f"Analyse le CV suivant:\n{cv_text}\n\nAnalyse complète du CV :"
        cv_analysis = generate_response(prompt)

        prompt = f"Analyse l'offre d'emploi suivante:\n{job_text}\n\nAnalyse complète de l'offre d'emploi :"
        job_analysis = generate_response(prompt)

        prompt = (f"Compare le CV suivant :\n{cv_text}\n\navec l'offre d'emploi suivante :\n{job_text}\n\n"
                  "Comparaison détaillée entre le CV et l'offre d'emploi :")
        comparison = generate_response(prompt)

        add_message(f"Analyse du CV: {cv_analysis}", "Bot")
        add_message(f"Analyse de l'offre d'emploi: {job_analysis}", "Bot")
        add_message(f"Comparaison: {comparison}", "Bot")
    except Exception as e:
        add_message(f"Une erreur est survenue : {str(e)}", "Bot")

# Fonction pour sélectionner les fichiers PDF et extraire leur texte
def select_pdfs(is_cv=True):
    pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    if pdf_paths:
        extracted_texts = extract_text_from_pdfs(pdf_paths)
        text_type = "CV" if is_cv else "offre d'emploi"
        for text in extracted_texts:
            add_message(f"Texte extrait du {text_type} PDF:\n{text}", "Bot")
            if is_cv:
                global cv_text
                cv_text = text
            else:
                global job_text
                job_text = text

# Création de l'interface utilisateur
root = tk.Tk()
root.title("Chatbot d'Analyse de CV et d'Offres d'Emploi")

chat_frame = tk.Frame(root)
chat_frame.pack(padx=10, pady=5, fill="both", expand=True)

chat_text = scrolledtext.ScrolledText(chat_frame, state=tk.DISABLED, wrap=tk.WORD)
chat_text.pack(padx=5, pady=5, fill="both", expand=True)

user_entry = tk.Entry(root, width=100)
user_entry.pack(padx=5, pady=5, fill="x")
user_entry.bind("<Return>", send_message)

button_frame = tk.Frame(root)
button_frame.pack(padx=10, pady=5, fill="x")

send_button = tk.Button(button_frame, text="Envoyer", command=send_message)
send_button.pack(side="left", padx=5)

cv_pdf_button = tk.Button(button_frame, text="Sélectionner des fichiers PDF pour les CV", command=lambda: select_pdfs(is_cv=True))
cv_pdf_button.pack(side="left", padx=5)

job_pdf_button = tk.Button(button_frame, text="Sélectionner des fichiers PDF pour les descriptions de postes", command=lambda: select_pdfs(is_cv=False))
job_pdf_button.pack(side="left", padx=5)

root.mainloop()
