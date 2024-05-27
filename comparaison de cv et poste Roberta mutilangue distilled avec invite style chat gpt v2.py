import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer XLM-RoBERTa
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base")

# Fonction pour analyser un CV
def analyze_cv(cv_text):
    inputs = tokenizer(cv_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    # Exemple d'analyse de base, à adapter selon votre cas d'utilisation
    analysis_results = {
        "score": scores.tolist()
    }
    return analysis_results

# Fonction pour analyser une offre d'emploi
def analyze_job_offer(job_offer_text):
    inputs = tokenizer(job_offer_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    scores = outputs.logits.softmax(dim=1)
    # Exemple d'analyse de base, à adapter selon votre cas d'utilisation
    analysis_results = {
        "score": scores.tolist()
    }
    return analysis_results

# Fonction pour comparer un CV avec une offre d'emploi
def compare_cv_job_offer(cv_analysis, job_offer_analysis):
    # Logique de comparaison basée sur les analyses effectuées
    comparison_results = {
        "comparison_score": (cv_analysis['score'], job_offer_analysis['score'])
    }
    return comparison_results

# Création de l'interface utilisateur avec Tkinter
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analyse des CV et des Offres d'Emploi")
        self.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Menu principal
        self.menu = tk.Frame(self)
        self.menu.pack(fill=tk.BOTH, expand=True)

        self.label = tk.Label(self.menu, text="Choisissez une option :", font=("Arial", 14))
        self.label.pack(pady=10)

        self.cv_button = tk.Button(self.menu, text="Analyser des CV", command=self.analyze_cv_ui)
        self.cv_button.pack(pady=5)

        self.job_offer_button = tk.Button(self.menu, text="Analyser une offre d'emploi", command=self.analyze_job_offer_ui)
        self.job_offer_button.pack(pady=5)

        self.compare_button = tk.Button(self.menu, text="Comparer le ou les CV et l'offre d'emploi", command=self.compare_ui)
        self.compare_button.pack(pady=5)

        self.result_text = scrolledtext.ScrolledText(self.menu, wrap=tk.WORD, width=80, height=20)
        self.result_text.pack(pady=20)

    def analyze_cv_ui(self):
        self.result_text.delete(1.0, tk.END)
        cv_text = self.get_text_from_file()
        if cv_text:
            results = analyze_cv(cv_text)
            self.result_text.insert(tk.END, f"Résultats de l'analyse du CV :\n{results}")

    def analyze_job_offer_ui(self):
        self.result_text.delete(1.0, tk.END)
        job_offer_text = self.get_text_from_file()
        if job_offer_text:
            results = analyze_job_offer(job_offer_text)
            self.result_text.insert(tk.END, f"Résultats de l'analyse de l'offre d'emploi :\n{results}")

    def compare_ui(self):
        self.result_text.delete(1.0, tk.END)
        cv_text = self.get_text_from_file(title="Sélectionnez un fichier de CV")
        job_offer_text = self.get_text_from_file(title="Sélectionnez un fichier d'offre d'emploi")
        if cv_text and job_offer_text:
            cv_results = analyze_cv(cv_text)
            job_offer_results = analyze_job_offer(job_offer_text)
            comparison_results = compare_cv_job_offer(cv_results, job_offer_results)
            self.result_text.insert(tk.END, f"Résultats de la comparaison :\n{comparison_results}")

    def get_text_from_file(self, title="Sélectionnez un fichier"):
        file_path = filedialog.askopenfilename(title=title, filetypes=(("Text files", "*.txt"), ("All files", "*.*")))
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        return None

# Exécution de l'application
if __name__ == "__main__":
    app = Application()
    app.mainloop()
