import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter.ttk import Progressbar
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import PyPDF2

# Charger le modèle et le tokenizer DistilRoBERTa multilingue
model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Fonction pour générer les embeddings
def generate_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Pré-traitement des données
def preprocess_text(text):
    return text.strip().lower()

# Fonction pour extraire le texte des fichiers PDF
def extract_text_from_pdfs(pdf_paths):
    texts = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            texts.append(preprocess_text(text))
    return texts

# Fonction pour afficher le menu principal
def show_menu():
    menu_text = ("Pour analyser un ou plusieurs CVs, une offre d'emploi ou comparer des CVs avec une offre d'emploi, "
                 "veuillez choisir une option du menu ci-dessous :\n- a) Analyser des CV\n- b) Analyser une offre d'emploi\n"
                 "- c) Comparer le ou les CV et l'offre d'emploi")
    menu_label.config(text=menu_text)

# Fonction pour lancer l'analyse des CV
def analyze_cvs():
    cv_texts = cv_text_area.get("1.0", tk.END).strip().split("\n\n")

    if not cv_texts[0] and cv_pdf_paths:
        cv_texts = extract_text_from_pdfs(cv_pdf_paths)

    if not cv_texts:
        messagebox.showerror("Erreur", "Veuillez fournir au moins un CV.")
        return

    # Analyse des CVs
    report = []
    for cv_text in cv_texts:
        report.append(analyze_single_cv(cv_text))

    display_report(report, "Analyse des CV")
    show_menu()

# Fonction pour analyser un CV unique
def analyze_single_cv(cv_text):
    # Simuler l'analyse détaillée en suivant la méthodologie fournie
    summary = {
        "Informations de contact": "Présentes",
        "Objectif professionnel": "Correspond",
        "Expérience professionnelle": "Pertinente",
        "Compétences techniques et non techniques": "Adéquates",
        "Réalisations": "Spécifiques",
        "Formation académique": "Pertinente",
        "Certifications": "Présentes",
        "Cohérence et chronologie": "Correcte",
        "Présentation et clarté": "Bonne",
        "Références": "Pertinentes",
        "Mots-clés": "Correspondants",
        "Adéquation culturelle": "Bonne",
        "Évolution de carrière": "Présente",
        "Activités et intérêts personnels": "Présents",
        "Métiers possibles": "Liste de métiers possibles avec synonymes"
    }
    return summary

# Fonction pour lancer l'analyse de l'offre d'emploi
def analyze_job():
    job_text = job_text_area.get("1.0", tk.END).strip()

    if not job_text and job_pdf_paths:
        job_text = extract_text_from_pdfs(job_pdf_paths)[0]

    if not job_text:
        messagebox.showerror("Erreur", "Veuillez fournir une description de poste.")
        return

    # Analyse de l'offre d'emploi
    report = analyze_single_job(job_text)
    display_report([report], "Analyse de l'offre d'emploi")
    show_menu()

# Fonction pour analyser une offre d'emploi unique
def analyze_single_job(job_text):
    # Simuler l'analyse détaillée en suivant la méthodologie fournie
    summary = {
        "Informations de l'entreprise": "Présentes",
        "Description du poste": "Identifiée",
        "Compétences requises": "Mentionnées",
        "Qualifications et expériences": "Correctes",
        "Critères de sélection": "Identifiés",
        "Avantages et conditions de travail": "Précisés",
        "Mots-clés": "Correspondants",
        "Titre du poste": "Approprié"
    }
    return summary

# Fonction pour lancer la comparaison entre CV et offre d'emploi
def compare_cv_and_job():
    cv_texts = cv_text_area.get("1.0", tk.END).strip().split("\n\n")
    job_text = job_text_area.get("1.0", tk.END).strip()

    if not cv_texts[0] and cv_pdf_paths:
        cv_texts = extract_text_from_pdfs(cv_pdf_paths)
    if not job_text and job_pdf_paths:
        job_text = extract_text_from_pdfs(job_pdf_paths)[0]

    if not cv_texts or not job_text:
        messagebox.showerror("Erreur", "Veuillez fournir au moins un CV et une description de poste.")
        return

    # Générer les embeddings pour les CV et l'offre d'emploi
    cv_embeddings = generate_embeddings(cv_texts)
    job_embedding = generate_embeddings([job_text])[0]

    # Générer le rapport de comparaison
    report = []
    for i, cv_text in enumerate(cv_texts):
        report.append(compare_single_cv_and_job(cv_text, job_text, cv_embeddings[i], job_embedding))

    display_report(report, "Comparaison des CV et de l'offre d'emploi")
    show_menu()

# Fonction pour comparer un CV unique et une offre d'emploi unique
def compare_single_cv_and_job(cv_text, job_text, cv_embedding, job_embedding):
    # Calcul de la similarité (méthode illustrée ici, mais peut être étendue)
    similarity_score = cosine_similarity(cv_embedding.unsqueeze(0), job_embedding.unsqueeze(0)).flatten()[0]

    # Points forts et faibles (exemple simplifié, à adapter selon le cas réel)
    points_forts = []
    points_faibles = []

    if similarity_score > 0.8:
        points_forts.append("Le CV est très aligné avec les exigences de l'offre d'emploi.")
    else:
        points_faibles.append("Le CV ne correspond pas pleinement aux exigences de l'offre d'emploi.")

    summary = {
        "CV": cv_text[:100] + "...",
        "Description de poste": job_text[:100] + "...",
        "Points forts": "; ".join(points_forts) if points_forts else "Aucun point fort notable",
        "Points faibles": "; ".join(points_faibles) if points_faibles else "Aucun point faible notable"
    }
    return summary

# Fonction pour afficher le rapport détaillé
def display_report(report, title):
    report_window = tk.Toplevel(root)
    report_window.title(title)

    report_text = scrolledtext.ScrolledText(report_window, width=100, height=30)
    report_text.pack(padx=10, pady=10)

    for entry in report:
        for key, value in entry.items():
            report_text.insert(tk.END, f"{key}: {value}\n")
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

menu_label = tk.Label(root, text="", wraplength=600)
menu_label.pack(padx=10, pady=10)
show_menu()

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

process_button_frame = tk.Frame(root)
process_button_frame.pack(pady=10)
process_cv_button = tk.Button(process_button_frame, text="Analyser des CV", command=analyze_cvs)
process_cv_button.pack(side="left", padx=5)
process_job_button = tk.Button(process_button_frame, text="Analyser une offre d'emploi", command=analyze_job)
process_job_button.pack(side="left", padx=5)
compare_button = tk.Button(process_button_frame, text="Comparer le ou les CV et l'offre d'emploi", command=compare_cv_and_job)
compare_button.pack(side="left", padx=5)

root.mainloop()
