import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar
import pandas as pd
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
def generate_response(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
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
                text += page.extract_text()
            texts.append(text.strip())
    return texts

# Fonction pour lancer le traitement
def process_data():
    cv_texts = cv_text_area.get("1.0", tk.END).strip().split("\n\n")
    job_texts = job_text_area.get("1.0", tk.END).strip().split("\n\n")

    if not cv_texts[0] and cv_pdf_paths:
        cv_texts = extract_text_from_pdfs(cv_pdf_paths)
    if not job_texts[0] and job_pdf_paths:
        job_texts = extract_text_from_pdfs(job_pdf_paths)

    if not cv_texts or not job_texts:
        messagebox.showerror("Erreur", "Veuillez fournir au moins un CV et une description de poste.")
        return

    report = []
    total_steps = len(cv_texts)
    for i, cv_text in enumerate(cv_texts):
        prompt = f"Analyse le CV suivant:\n{cv_text}\n\nAnalyse complète du CV :"
        cv_analysis = generate_response(prompt)

        prompt = f"Analyse l'offre d'emploi suivante:\n{job_texts[0]}\n\nAnalyse complète de l'offre d'emploi :"
        job_analysis = generate_response(prompt)

        prompt = (f"Compare le CV suivant :\n{cv_text}\n\navec l'offre d'emploi suivante :\n{job_texts[0]}\n\n"
                  "Comparaison détaillée entre le CV et l'offre d'emploi :")
        comparison = generate_response(prompt)

        report.append({
            "CV": cv_text[:30] + "...",
            "Analyse du CV": cv_analysis,
            "Analyse de l'offre d'emploi": job_analysis,
            "Comparaison": comparison
        })
        update_progress(i + 1, total_steps)

    display_report(report)

# Fonction pour mettre à jour la barre de progression
def update_progress(current_step, total_steps):
    progress = int((current_step / total_steps) * 100)
    progress_var.set(progress)
    root.update_idletasks()

# Fonction pour afficher le rapport détaillé
def display_report(report):
    report_window = tk.Toplevel(root)
    report_window.title("Rapport d'analyse de correspondance")

    report_text = scrolledtext.ScrolledText(report_window, width=100, height=30)
    report_text.pack(padx=10, pady=10)

    for entry in report:
        report_text.insert(tk.END, f"CV: {entry['CV']}\n")
        report_text.insert(tk.END, f"Analyse du CV: {entry['Analyse du CV']}\n")
        report_text.insert(tk.END, f"Analyse de l'offre d'emploi: {entry['Analyse de l'offre d'emploi']}\n")
        report_text.insert(tk.END, f"Comparaison: {entry['Comparaison']}\n")
        report_text.insert(tk.END, "-"*100 + "\n")

    messagebox.showinfo("Succès", "Le rapport a été généré et affiché.")

# Fonctions pour sélectionner les fichiers PDF
def select_cv_pdfs():
    global cv_pdf_paths
    cv_pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    cv_pdf_label.config(text=f"{len(cv_pdf_paths)} fichiers sélectionnés")

def select_job_pdfs():
    global job_pdf_paths
    job_pdf_paths = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
    job_pdf_label.config(text=f"{len(job_pdf_paths)} fichiers sélectionnés")

# Création de l'interface utilisateur
root = tk.Tk()
root.title("Analyse de CV et d'Offres d'Emploi")

cv_frame = tk.Frame(root)
cv_frame.pack(padx=10, pady=5, fill="x")
cv_button = tk.Button(cv_frame, text="Sélectionner des fichiers PDF pour les CV", command=select_cv_pdfs)
cv_button.pack(side="left")
cv_pdf_label = tk.Label(cv_frame, text="")
cv_pdf_label.pack(side="left", padx=5)
cv_text_area = scrolledtext.ScrolledText(cv_frame, width=60, height=10)
cv_text_area.pack(padx=5, pady=5)
cv_text_area.insert(tk.END, "Copiez et collez le texte des CV ici, ou sélectionnez des fichiers PDF.")

job_frame = tk.Frame(root)
job_frame.pack(padx=10, pady=5, fill="x")
job_button = tk.Button(job_frame, text="Sélectionner des fichiers PDF pour les descriptions de postes", command=select_job_pdfs)
job_button.pack(side="left")
job_pdf_label = tk.Label(job_frame, text="")
job_pdf_label.pack(side="left", padx=5)
job_text_area = scrolledtext.ScrolledText(job_frame, width=60, height=10)
job_text_area.pack(padx=5, pady=5)
job_text_area.insert(tk.END, "Copiez et collez le texte des descriptions de postes ici, ou sélectionnez des fichiers PDF.")

progress_var = tk.IntVar()
progress_bar = Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(pady=10, fill="x")

process_button = tk.Button(root, text="Lancer le traitement", command=process_data)
process_button.pack(pady=10)

root.mainloop()
