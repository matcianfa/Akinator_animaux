"""
API FastAPI pour Qui est-ce avec Google Drive
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
import csv
import io
import os
import json
import uuid
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from dotenv import load_dotenv
load_dotenv()

# ── Constantes ──────────────────────────────────────────────────────────────
REPONSES_POSSIBLES = ["Oui", "Plutôt oui", "Je ne sais pas (ou entre oui et non)", "Plutôt non", "Non"]
VALEURS_REPONSES   = [1, 0.75, 0.5, 0.25, 0]
SEUIL              = 0.7   # seuil Akinator classique (conservé)
SEUIL_QUI_EST_CE   = 0.90  # seuil pour la proposition dans Qui est-ce

# ── Configuration Google Drive ───────────────────────────────────────────────
SCOPES                  = ['https://www.googleapis.com/auth/drive']
GDRIVE_FILE_NAME        = "qui_est_ce_animaux.csv"   # ← fichier renommé
GDRIVE_SUGGESTIONS_FILE = "contenu_a_ajouter.csv"

# Stockage des sessions en mémoire
sessions: Dict[str, dict] = {}

# ── Stockage des sessions version classique ───────────────────────────────────
from itertools import combinations
import random as _random

CSV_CLASSIQUE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qui_est_ce_classique.csv")
sessions_classique: Dict[str, dict] = {}

def charger_donnees_classique():
    if not os.path.exists(CSV_CLASSIQUE_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Fichier '{CSV_CLASSIQUE_PATH}' introuvable. Placez-le dans le même répertoire que main.py."
        )
    with open(CSV_CLASSIQUE_PATH, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        personnages = header[1:]
        attributs, donnees = [], []
        for row in reader:
            if not row or not row[0].strip():
                continue
            attributs.append(row[0])
            donnees.append([int(v) for v in row[1:]])
    return personnages, attributs, donnees

def meilleure_combinaison_classique(donnees, remaining, attrs_disponibles, max_size=3):
    n = len(remaining)
    if n == 0:
        return []
    attrs_list = list(attrs_disponibles)
    _random.shuffle(attrs_list)          # ordre aléatoire → variété à score égal
    effective_max = min(max_size, len(attrs_list))
    if n <= 4:
        effective_max = min(2, effective_max)
    best_combo, best_score = None, float('inf')
    for size in range(1, effective_max + 1):
        for combo in combinations(attrs_list, size):
            count_yes = sum(1 for c in remaining if any(donnees[a][c] == 1 for a in combo))
            score = abs(count_yes - n / 2)
            if score < best_score:
                best_score = score
                best_combo = list(combo)
            if best_score == 0:
                break
        if best_score == 0:
            break
    return best_combo if best_combo is not None else [attrs_list[0]]

def formater_question_classique(attributs, attr_indices):
    textes = [attributs[a] for a in attr_indices]
    return "Le personnage " + " ou ".join(textes) + " ?"

# ── Application ──────────────────────────────────────────────────────────────
app = FastAPI(title="Qui est-ce API")

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modèles Pydantic ─────────────────────────────────────────────────────────
# (anciens modèles conservés pour compatibilité)
class SessionResponse(BaseModel):
    session_id: str
    question: str
    question_number: int
    reponses_possibles: list = []
    class Config:
        arbitrary_types_allowed = True

class AnswerRequest(BaseModel):
    session_id: str
    reponse: int

class GuessResponse(BaseModel):
    animal: str
    probabilite: float
    is_final: bool
    question: Optional[str] = None
    question_number: Optional[int] = None

class ConfirmRequest(BaseModel):
    session_id: str
    correct: bool

class SuggestionRequest(BaseModel):
    session_id: str
    animal: str
    question: str

# Nouveaux modèles pour Qui est-ce
class StartQECRequest(BaseModel):
    animal_joueur_index: int = -1

class PlayerAsksRequest(BaseModel):
    session_id: str
    question_index: int

class PlayerAnswersAkinatorRequest(BaseModel):
    session_id: str
    reponse: int   # index dans VALEURS_REPONSES (0-4)

class PlayerProposesRequest(BaseModel):
    session_id: str
    animal_index: int

class ConfirmAkinatorProposalRequest(BaseModel):
    session_id: str
    correct: bool

class DeclareJoueurAnimalRequest(BaseModel):
    session_id: str
    animal_index: int

# ── Fonctions Google Drive (inchangées) ──────────────────────────────────────
def get_drive_service():
    try:
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        if not creds_json:
            raise ValueError("GOOGLE_CREDENTIALS non définie")
        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Google Drive: {str(e)}")

def find_file_id(service, filename):
    try:
        results = service.files().list(
            q=f"name='{filename}' and trashed=false",
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        files = results.get('files', [])
        return files[0]['id'] if files else None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur recherche fichier: {str(e)}")

def download_csv_from_drive():
    try:
        service  = get_drive_service()
        file_id  = find_file_id(service, GDRIVE_FILE_NAME)
        if not file_id:
            raise HTTPException(status_code=404, detail=f"Fichier '{GDRIVE_FILE_NAME}' non trouvé sur Google Drive")
        request     = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()
        downloader  = MediaIoBaseDownload(file_handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        file_handle.seek(0)
        return file_handle.read().decode('utf-8')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur téléchargement: {str(e)}")

def upload_csv_to_drive(content, filename=GDRIVE_FILE_NAME):
    try:
        service      = get_drive_service()
        file_id      = find_file_id(service, filename)
        file_metadata = {'name': filename}
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode('utf-8')),
            mimetype='text/csv',
            resumable=True
        )
        if file_id:
            service.files().update(fileId=file_id, media_body=media).execute()
        else:
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

def charger_donnees():
    try:
        csv_content = download_csv_from_drive()
        reader      = csv.reader(io.StringIO(csv_content))
        animaux     = next(reader)[1:]
        compteur_apparitions = [int(val) for val in next(reader)[1:]]
        questions, donnees = [], []
        for ligne in reader:
            questions.append(ligne[0])
            donnees.append([float(val) for val in ligne[1:]])
        donnees = np.asarray(donnees)
        return animaux, compteur_apparitions, questions, donnees
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement données: {str(e)}")

def sauvegarde_csv(animaux, compteur_apparitions, questions, donnees):
    try:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Question"] + animaux)
        writer.writerow(["Compteur d'apparitions"] + compteur_apparitions)
        for i in range(len(questions)):
            writer.writerow([questions[i]] + [f"{v:.3f}" for v in donnees[i]])
        upload_csv_to_drive(output.getvalue(), GDRIVE_FILE_NAME)
    except Exception as e:
        print(f"Erreur sauvegarde: {str(e)}")

def sauvegarder_suggestion(animal, question, reponses_donnees, questions):
    try:
        service = get_drive_service()
        file_id = find_file_id(service, GDRIVE_SUGGESTIONS_FILE)
        lignes_existantes = []
        if file_id:
            try:
                request     = service.files().get_media(fileId=file_id)
                file_handle = io.BytesIO()
                downloader  = MediaIoBaseDownload(file_handle, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                file_handle.seek(0)
                lignes_existantes = list(csv.reader(io.StringIO(file_handle.read().decode('utf-8'))))
            except:
                pass
        output = io.StringIO()
        writer = csv.writer(output)
        if not lignes_existantes:
            writer.writerow(["Animal", "Question proposée"])
        else:
            for ligne in lignes_existantes:
                writer.writerow(ligne)
        writer.writerow([animal, question])
        upload_csv_to_drive(output.getvalue(), GDRIVE_SUGGESTIONS_FILE)
        return True
    except:
        return False

# ── Fonctions de calcul (inchangées) ─────────────────────────────────────────
def donner_proba_animaux_sachant_r(r, donnees, proba_animaux):
    numerateurs  = np.maximum(1 - abs(r - donnees), 0.05) * proba_animaux
    denominateur = np.sum(numerateurs, axis=1)
    denominateur[denominateur == 0] = 1
    return np.divide(numerateurs, denominateur.reshape(-1, 1)), denominateur

def calcul_IM(donnees, proba_animaux):
    IM = 0
    for r in VALEURS_REPONSES:
        proba_animaux_sachant_r, p_r = donner_proba_animaux_sachant_r(r, donnees, proba_animaux)
        h_r = np.sum(
            np.where(proba_animaux_sachant_r > 0,
                     -proba_animaux_sachant_r * np.log2(proba_animaux_sachant_r), 0),
            axis=1
        )
        IM += h_r * p_r
    return IM

def choix_meilleure_question(donnees, proba_animaux, questions_pas_encore_posees):
    IM = calcul_IM(donnees, proba_animaux)
    if any(questions_pas_encore_posees):
        return np.argmin(np.where(questions_pas_encore_posees, IM, np.inf))
    return None

def recherche_bonne_reponse(proba_animaux, animaux):
    """Seuil classique (0.7) – conservé pour l'ancien mode Akinator."""
    i_premier, *_ = np.argsort(proba_animaux)[::-1]
    if proba_animaux[i_premier] > SEUIL:
        return i_premier, proba_animaux[i_premier]
    return None, None

def recherche_bonne_reponse_qec(proba_animaux):
    """Seuil élevé (0.90) pour la proposition dans Qui est-ce."""
    i_premier = int(np.argmax(proba_animaux))
    if proba_animaux[i_premier] > SEUIL_QUI_EST_CE:
        return i_premier, float(proba_animaux[i_premier])
    return None, None

def actualisation_valeurs_theoriques(donnees, indice_animal, reponses_donnees, alpha=0.1):
    for question, reponse in reponses_donnees.items():
        donnees[question, indice_animal] += alpha * (VALEURS_REPONSES[reponse] - donnees[question, indice_animal])

def valeur_vers_reponse_index(valeur: float) -> int:
    """
    Convertit une valeur numérique CSV en index de réponse (0-4).
    Les seuils sont les points médians entre les valeurs discrètes.
    """
    seuils = [0.875, 0.625, 0.375, 0.125]
    for i, seuil in enumerate(seuils):
        if valeur >= seuil:
            return i
    return 4

def terminer_partie_et_apprendre(session: dict):
    """Sauvegarde l'apprentissage à partir des réponses du joueur sur son animal."""
    animal_joueur = session.get("animal_joueur", -1)
    if animal_joueur == -1 or not session.get("reponses_donnees_akinator"):
        return  # le joueur n'avait pas sélectionné d'animal → pas d'apprentissage
    session["compteur_apparitions"][animal_joueur] += 1
    actualisation_valeurs_theoriques(
        session["donnees"],
        animal_joueur,
        session["reponses_donnees_akinator"]
    )
    sauvegarde_csv(
        session["animaux"],
        session["compteur_apparitions"],
        session["questions"],
        session["donnees"]
    )

# ── Interface HTML ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def get_interface():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qui est-ce ? 🎭</title>
<style>
  :root {
    --primary: #667eea;
    --secondary: #764ba2;
    --accent: #f5576c;
    --green: #43b89c;
    --bg: #f0f2ff;
    --card-bg: #ffffff;
    --radius: 14px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: var(--bg);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 20px;
  }
  h1 { color: var(--secondary); font-size: 2.2em; margin-bottom: 6px; }
  h2 { color: var(--secondary); font-size: 1.6em; margin-bottom: 10px; }
  h3 { color: var(--primary); font-size: 1.1em; margin-bottom: 8px; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 1em; }

  /* ── Screens ── */
  .screen { display:none; width:100%; max-width:860px; animation: fadeIn .3s; }
  .screen.active { display:block; }
  @keyframes fadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

  /* ── Card ── */
  .card {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 6px 28px rgba(102,126,234,.15);
    padding: 32px;
    margin-bottom: 20px;
  }

  /* ── Animal grid ── */
  .animals-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 10px;
    margin: 20px 0;
  }
  .animal-card {
    background: var(--bg);
    border: 2px solid transparent;
    border-radius: 10px;
    padding: 12px 8px;
    cursor: pointer;
    text-align: center;
    transition: all .2s;
    font-size: .95em;
    font-weight: 500;
    color: #444;
  }
  .animal-card:hover { border-color: var(--primary); background: #e8ecff; }
  .animal-card.selected { border-color: var(--secondary); background: #e0d9f7; color: var(--secondary); font-weight: 700; }
  .animal-card .emoji { font-size: 1.8em; display:block; margin-bottom: 4px; }

  /* ── Buttons ── */
  button {
    padding: 12px 24px; font-size: 1em; border: none; border-radius: 10px;
    cursor: pointer; transition: all .2s; font-weight: 600; color: white;
  }
  .btn-primary { background: linear-gradient(135deg, var(--primary), var(--secondary)); }
  .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(102,126,234,.4); }
  .btn-primary:disabled { opacity: .5; cursor:not-allowed; transform:none; box-shadow:none; }
  .btn-danger { background: linear-gradient(135deg, #f093fb, var(--accent)); }
  .btn-danger:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(245,87,108,.4); }
  .btn-success { background: linear-gradient(135deg, #43b89c, #2ecc71); }
  .btn-success:hover { transform: translateY(-2px); }
  .btn-outline {
    background: white; color: var(--primary);
    border: 2px solid var(--primary);
  }
  .btn-outline:hover { background: #e8ecff; }
  .btn-answer {
    width:100%; margin-bottom:8px;
    background: var(--primary); text-align:left; padding:12px 18px;
  }
  .btn-answer:hover { background: var(--secondary); transform: translateX(4px); }

  /* ── Turn banner ── */
  .turn-banner {
    border-radius: var(--radius);
    padding: 14px 24px;
    font-size: 1.1em;
    font-weight: 700;
    color: white;
    margin-bottom: 16px;
    text-align: center;
  }
  .turn-joueur  { background: linear-gradient(135deg, var(--primary), var(--secondary)); }
  .turn-akinator { background: linear-gradient(135deg, #f093fb, var(--accent)); }

  /* ── Game layout ── */
  .game-layout {
    display: grid;
    grid-template-columns: 1fr 1.6fr;
    gap: 16px;
  }
  @media(max-width:640px){ .game-layout{grid-template-columns:1fr;} }

  /* ── History ── */
  .history-panel {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 4px 18px rgba(102,126,234,.1);
    padding: 20px;
    max-height: 460px;
    overflow-y: auto;
  }
  .history-item {
    border-left: 3px solid var(--primary);
    padding: 8px 10px;
    margin-bottom: 10px;
    background: var(--bg);
    border-radius: 0 8px 8px 0;
    font-size: .9em;
  }
  .history-item.akinator-asks { border-color: var(--accent); }
  .history-q { color: #555; margin-bottom:3px; }
  .history-a { font-weight:700; color: var(--secondary); }
  .history-a.akinator-ans { color: var(--accent); }

  /* ── Action panel ── */
  .action-panel {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 4px 18px rgba(102,126,234,.1);
    padding: 24px;
  }
  select {
    width:100%; padding:12px; border-radius:10px;
    border: 2px solid #dde; font-size:.95em;
    margin-bottom: 14px; background: var(--bg); color: #333;
    appearance: none; cursor:pointer;
  }
  select:focus { outline:none; border-color: var(--primary); }

  /* ── Akinator answer bubble ── */
  .answer-bubble {
    background: linear-gradient(135deg, #f093fb22, #f5576c11);
    border: 2px solid var(--accent);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 1.1em;
    color: var(--accent);
    font-weight: 700;
    text-align: center;
  }

  /* ── Proposal box ── */
  .proposal-box {
    background: linear-gradient(135deg, #f093fb, var(--accent));
    color: white;
    border-radius: var(--radius);
    padding: 24px;
    text-align: center;
    margin-bottom: 14px;
  }
  .proposal-animal { font-size: 1.8em; font-weight:800; margin: 10px 0; }
  .proposal-proba  { font-size: .95em; opacity:.85; margin-bottom: 16px; }

  /* ── Modal ── */
  .modal-overlay {
    display:none; position:fixed; inset:0;
    background:rgba(0,0,0,.5); z-index:100;
    align-items:center; justify-content:center;
  }
  .modal-overlay.active { display:flex; }
  .modal-box {
    background: white; border-radius: 18px;
    padding: 28px; max-width: 500px; width:90%;
    max-height: 80vh; overflow-y:auto;
  }
  .modal-box h3 { margin-bottom: 16px; }
  .modal-grid {
    display:grid; grid-template-columns: repeat(auto-fill,minmax(110px,1fr));
    gap:8px; margin-bottom:16px;
  }
  .modal-animal {
    background: var(--bg); border:2px solid transparent;
    border-radius:8px; padding:10px 6px; text-align:center;
    cursor:pointer; font-size:.9em; font-weight:500; color:#444;
    transition: all .15s;
  }
  .modal-animal:hover { border-color: var(--accent); background:#ffe0e8; }
  .modal-animal.selected { border-color: var(--accent); background:#f5576c22; font-weight:700; color:var(--accent); }
  .modal-animal .emoji { font-size:1.5em; display:block; margin-bottom:3px; }
  .modal-actions { display:flex; gap:10px; justify-content:flex-end; }

  /* ── End screen ── */
  .end-box {
    text-align:center;
    padding: 40px 24px;
  }
  .end-title { font-size:2em; margin-bottom:16px; }
  .reveal-row {
    display:flex; justify-content:center; gap:24px;
    flex-wrap:wrap; margin: 24px 0;
  }
  .reveal-card {
    background: var(--bg); border-radius: 14px;
    padding: 20px 28px; min-width:150px; text-align:center;
  }
  .reveal-card .label { color:#888; font-size:.85em; margin-bottom:6px; }
  .reveal-card .name  { font-size:1.3em; font-weight:800; color:var(--secondary); }
  .reveal-card .big-emoji { font-size:2.5em; }

  /* ── Spinner ── */
  .loading-overlay { display:none; position:fixed; inset:0;
    background:rgba(255,255,255,.75); z-index:200;
    align-items:center; justify-content:center; flex-direction:column; }
  .loading-overlay.active { display:flex; }
  .spinner {
    border: 4px solid #dde; border-top-color: var(--primary);
    border-radius:50%; width:42px; height:42px;
    animation: spin 1s linear infinite; margin-bottom:12px;
  }
  @keyframes spin { to{transform:rotate(360deg)} }

  .mt { margin-top:12px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; }

  /* ── Bouton technique fixe ── */
  #btnAdmin {
    position: fixed; top: 14px; left: 14px; z-index: 300;
    background: #fff; color: var(--secondary);
    border: 2px solid var(--secondary); border-radius: 10px;
    padding: 8px 14px; font-size: .85em; font-weight: 700;
    cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,.12);
    transition: all .2s;
  }
  #btnAdmin:hover { background: var(--secondary); color: white; }

  /* ── Panneau admin ── */
  #adminPanel {
    display: none; position: fixed; inset: 0; z-index: 400;
    background: rgba(0,0,0,.5); align-items: flex-start;
    justify-content: center; padding-top: 60px; overflow-y: auto;
  }
  #adminPanel.active { display: flex; }
  #adminBox {
    background: white; border-radius: 18px;
    padding: 28px; width: 95%; max-width: 900px;
    margin-bottom: 40px; box-shadow: 0 12px 40px rgba(0,0,0,.25);
  }
  #adminBox h2 { color: var(--secondary); margin-bottom: 6px; }
  .admin-section {
    background: var(--bg); border-radius: 12px;
    padding: 20px; margin-top: 20px;
  }
  .admin-section h3 { margin-bottom: 14px; }

  /* Tableau des questions existantes */
  .q-table { width:100%; border-collapse: collapse; font-size:.88em; }
  .q-table th { background: var(--primary); color:white; padding:8px 10px; text-align:left; }
  .q-table td { padding:7px 10px; border-bottom:1px solid #eee; vertical-align:middle; }
  .q-table tr:last-child td { border-bottom: none; }
  .q-table tr.clickable { cursor:pointer; transition:background .15s; }
  .q-table tr.clickable:hover td { background:#eef0ff; }
  .q-table tr.selected-row td { background:#dde0ff; font-weight:700; color:var(--secondary); }

  /* Bouton supprimer */
  .btn-delete {
    background: linear-gradient(135deg,#e74c3c,#c0392b);
    color:white; padding:10px 18px; font-size:.9em;
  }
  .btn-delete:hover { opacity:.88; transform:translateY(-1px); }

  /* Badge mode édition */
  .edit-badge {
    display:inline-block; background:var(--secondary); color:white;
    border-radius:20px; padding:3px 12px; font-size:.8em;
    font-weight:700; margin-left:8px; vertical-align:middle;
  }

  /* Grille de réponses par animal */
  .animal-answer-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 10px; margin-top: 14px;
  }
  .animal-answer-row {
    background: white; border-radius: 10px;
    padding: 10px 12px; display:flex; flex-direction:column; gap:6px;
    box-shadow: 0 1px 6px rgba(0,0,0,.07);
  }
  .animal-answer-row .a-name {
    font-weight: 700; font-size:.9em; color: var(--secondary);
    display:flex; align-items:center; gap:5px;
  }
  .animal-answer-row select {
    margin-bottom: 0; font-size: .85em; padding: 7px 10px;
  }
  .admin-input {
    width:100%; padding:12px; border-radius:10px;
    border: 2px solid #dde; font-size:1em;
    background: white; color: #333; margin-bottom:14px;
  }
  .admin-input:focus { outline:none; border-color: var(--primary); }

  /* ── Panneau technique CLASSIQUE ── */
  #adminClassiquePanel {
    display: none; position: fixed; inset: 0; z-index: 400;
    background: rgba(0,0,0,.5); align-items: flex-start;
    justify-content: center; padding-top: 60px; overflow-y: auto;
  }
  #adminClassiquePanel.active { display: flex; }
  #adminClassiqueBox {
    background: white; border-radius: 18px; padding: 28px;
    width: 95%; max-width: 960px; margin-bottom: 40px;
    box-shadow: 0 12px 40px rgba(0,0,0,.25);
  }
  #adminClassiqueBox h2 { color: #1e7a55; margin-bottom: 6px; }
  .cl-admin-perso-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 8px; margin: 14px 0;
  }
  .cl-admin-perso-card {
    background: #f0fff8; border: 2px solid #b2e8d4; border-radius: 10px;
    padding: 10px 6px; text-align: center; cursor: pointer;
    font-size: .88em; font-weight: 600; color: #1e7a55; transition: all .15s;
  }
  .cl-admin-perso-card:hover { border-color: #43b89c; background: #c8f5e4; }
  .cl-admin-perso-card.cl-admin-selected {
    border-color: #1e7a55; background: #a0e8cc; box-shadow: 0 0 0 3px #43b89c44;
  }
  .cl-admin-attr-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
    gap: 8px; margin: 14px 0;
  }
  .cl-admin-attr-item {
    display: flex; align-items: center; gap: 10px;
    border-radius: 10px; padding: 10px 14px; cursor: pointer;
    transition: all .15s; font-size: .9em; font-weight: 600;
    border: 2px solid #b2e8d4;
  }
  .cl-admin-attr-item.val-1 { background: #c8f5e4; border-color: #43b89c; color: #1e7a55; }
  .cl-admin-attr-item.val-0 { background: #fff0f0; border-color: #f5a0a0; color: #b03030; }
  .cl-admin-attr-badge {
    min-width: 26px; height: 26px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-weight: 900;
  }
  .val-1 .cl-admin-attr-badge { background: #43b89c; color: white; }
  .val-0 .cl-admin-attr-badge { background: #e74c3c; color: white; }
  .cl-admin-section {
    background: #f8fffe; border-radius: 12px; padding: 18px; margin-top: 18px;
    border: 1px solid #d0f0e4;
  }
  .cl-admin-section h3 { color: #1e7a55; margin-bottom: 10px; }

  /* ═══════════════════════════════════════
     BOUTON VERSION CLASSIQUE
  ═══════════════════════════════════════ */
  #btnClassique {
    position: fixed; top: 54px; left: 14px; z-index: 300;
    background: #fff; color: #43b89c;
    border: 2px solid #43b89c; border-radius: 10px;
    padding: 8px 14px; font-size: .85em; font-weight: 700;
    cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,.12);
    transition: all .2s;
  }
  #btnClassique:hover { background: #43b89c; color: white; }
  #btnClassique.active-mode { background: #43b89c; color: white; }

  /* ═══════════════════════════════════════
     VERSION CLASSIQUE — STYLES
  ═══════════════════════════════════════ */

  /* Grille des personnages (sélection + jeu) */
  .cl-perso-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 8px;
    margin: 14px 0;
  }
  .cl-perso-card {
    background: var(--bg);
    border: 2px solid transparent;
    border-radius: 10px;
    padding: 10px 6px;
    text-align: center;
    cursor: pointer;
    font-size: .85em;
    font-weight: 600;
    color: #444;
    transition: all .2s;
    position: relative;
    user-select: none;
  }
  .cl-perso-card:hover { border-color: var(--green); background: #e0f5f0; }
  .cl-perso-card.selected { border-color: var(--green); background: #c8ede7; color: #2a7a68; font-weight: 800; }
  .cl-perso-card.eliminated {
    opacity: .25;
    filter: grayscale(1);
    pointer-events: none;
  }
  .cl-perso-card.manually-eliminated {
    opacity: .18;
    filter: grayscale(1) blur(1px);
    pointer-events: auto;   /* peut être rétabli */
  }
  .cl-perso-card .cl-x-badge {
    position: absolute; top: 2px; right: 4px;
    font-size: .75em; color: #e74c3c; font-weight: 900;
  }

  /* Attributs avec cases à cocher */
  .cl-attr-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 340px;
    overflow-y: auto;
    padding-right: 4px;
    margin-bottom: 12px;
  }
  .cl-attr-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 12px;
    border-radius: 8px;
    background: var(--bg);
    cursor: pointer;
    transition: background .15s;
    font-size: .92em;
  }
  .cl-attr-item:hover { background: #e0d9f7; }
  .cl-attr-item.checked { background: #dde0ff; font-weight: 600; color: var(--secondary); }
  .cl-attr-item input[type=checkbox] {
    width: 17px; height: 17px; cursor: pointer; accent-color: var(--secondary);
    flex-shrink: 0;
  }

  /* Prévisualisation de la question */
  .cl-question-preview {
    background: linear-gradient(135deg, #e8ecff, #f0e8ff);
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 1em;
    font-weight: 600;
    color: var(--secondary);
    min-height: 50px;
    transition: all .2s;
  }

  /* Réponse Oui/Non d'Akinator (version classique) */
  .cl-answer-bubble-oui {
    background: linear-gradient(135deg, #43b89c22, #2ecc7122);
    border: 2px solid var(--green);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 1.2em;
    color: #27ae60;
    font-weight: 800;
    text-align: center;
  }
  .cl-answer-bubble-non {
    background: linear-gradient(135deg, #f5576c22, #f09afb22);
    border: 2px solid var(--accent);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 14px 0;
    font-size: 1.2em;
    color: var(--accent);
    font-weight: 800;
    text-align: center;
  }

  /* Panneau "plateau" du joueur (mini grille) */
  .cl-board-panel {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: 0 4px 18px rgba(102,126,234,.1);
    padding: 14px;
  }
  .cl-board-panel h3 { font-size: .95em; margin-bottom: 8px; }
  .cl-board-hint {
    font-size: .75em;
    color: #aaa;
    margin-bottom: 8px;
    font-style: italic;
  }

  /* Bouton Oui / Non classique */
  .btn-oui { background: linear-gradient(135deg, #43b89c, #2ecc71); }
  .btn-oui:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(67,184,156,.4); }
  .btn-non { background: linear-gradient(135deg, #f093fb, var(--accent)); }
  .btn-non:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(245,87,108,.4); }

  /* Compteur de candidats restants */
  .cl-remaining-badge {
    display: inline-block;
    background: var(--secondary);
    color: white;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .8em;
    font-weight: 700;
    margin-left: 8px;
  }

  /* Modal proposition classique */
  #clModalProposition .modal-box { max-width: 620px; }
</style>
</head>
<body>

<!-- ── Bouton Technique ── -->
<button id="btnAdmin" onclick="toggleAdmin()">⚙️ Technique</button>
<!-- ── Bouton Version Classique ── -->
<button id="btnClassique" onclick="switchMode()">🧩 Version Classique</button>

<!-- ══════════════════════════════════════
     PANNEAU TECHNIQUE (ADMIN)
══════════════════════════════════════════ -->
<div id="adminPanel">
  <div id="adminBox">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <h2>⚙️ Panneau technique</h2>
      <button class="btn-outline" onclick="toggleAdmin()" style="padding:8px 16px;font-size:.9em">
        ✕ Fermer
      </button>
    </div>
    <p style="color:#888;font-size:.9em">
      Gérez les questions du fichier <code>qui_est_ce_animaux.csv</code> sur Google Drive.
    </p>

    <!-- Liste des questions existantes -->
    <div class="admin-section">
      <h3>📋 Questions existantes
        <span style="color:#888;font-size:.8em;font-weight:400;margin-left:6px">
          — cliquez sur une ligne pour la modifier
        </span>
      </h3>
      <div id="adminQuestionsTable" style="overflow-x:auto">
        <p style="color:#aaa">Chargement…</p>
      </div>
    </div>

    <!-- Section modification/suppression (cachée par défaut) -->
    <div class="admin-section" id="editSection" style="display:none;border:2px solid var(--secondary);">
      <h3>
        ✏️ Modifier la question
        <span class="edit-badge" id="editBadge">#?</span>
      </h3>
      <input
        class="admin-input"
        type="text"
        id="editQuestionText"
        placeholder="Texte de la question…"
        maxlength="120"
      />
      <p style="color:#666;font-size:.9em;margin-bottom:12px">
        Modifiez les valeurs par animal :
      </p>
      <div class="animal-answer-grid" id="editAnimalGrid"></div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:18px;flex-wrap:wrap;gap:10px;">
        <button class="btn-delete" onclick="supprimerQuestion()">
          🗑️ Supprimer cette question
        </button>
        <div class="row" style="justify-content:flex-end">
          <button class="btn-outline" onclick="annulerModifications()">
            🔄 Annuler les modifications
          </button>
          <button class="btn-primary" onclick="sauvegarderModification()">
            💾 Sauvegarder les modifications
          </button>
        </div>
      </div>
      <div id="editSaveMsg" style="margin-top:12px;display:none;"></div>
    </div>

    <!-- Ajout d'une nouvelle question -->
    <div class="admin-section">
      <h3>➕ Ajouter une nouvelle question</h3>
      <input
        class="admin-input"
        type="text"
        id="newQuestionText"
        placeholder="Ex : A-t-il des plumes ?"
        maxlength="120"
      />
      <p style="color:#666;font-size:.9em;margin-bottom:12px">
        Répondez pour chaque animal (la valeur sera stockée dans le CSV) :
      </p>
      <div class="animal-answer-grid" id="animalAnswerGrid">
        <p style="color:#aaa">Chargement…</p>
      </div>
      <div style="text-align:right;margin-top:18px">
        <button class="btn-primary" onclick="sauvegarderNouvelleQuestion()">
          💾 Sauvegarder la question
        </button>
      </div>
      <div id="adminSaveMsg" style="margin-top:12px;display:none;"></div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════
     PANNEAU TECHNIQUE CLASSIQUE
══════════════════════════════════════════ -->
<div id="adminClassiquePanel">
  <div id="adminClassiqueBox">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
      <h2>⚙️ Technique — Version Classique</h2>
      <button class="btn-outline" onclick="toggleAdminClassique()"
              style="padding:8px 16px;font-size:.9em;color:#1e7a55;border-color:#1e7a55;">
        ✕ Fermer
      </button>
    </div>
    <p style="color:#888;font-size:.9em">
      Cliquez sur un personnage pour modifier ses attributs ou son nom.
      Le fichier <code>qui_est_ce_classique.csv</code> sera mis à jour.
    </p>

    <div class="cl-admin-section">
      <h3>🎭 Personnages — cliquez pour modifier</h3>
      <div id="clAdminPersoGrid" class="cl-admin-perso-grid">
        <p style="color:#aaa">Chargement…</p>
      </div>
    </div>

    <div class="cl-admin-section" id="clAdminEditSection"
         style="display:none;border:2px solid #43b89c;">
      <h3>✏️ Modifier :
        <span id="clAdminEditBadge" style="color:#43b89c;font-weight:800;"></span>
      </h3>

      <div style="margin-bottom:16px;">
        <label style="font-weight:700;font-size:.9em;color:#555;display:block;margin-bottom:6px;">
          Nom du personnage :
        </label>
        <input type="text" id="clAdminNomInput" class="admin-input"
               placeholder="Nom du personnage…" maxlength="40"
               style="border-color:#43b89c;max-width:340px;" />
      </div>

      <p style="font-weight:700;font-size:.9em;color:#555;margin-bottom:8px;">
        Attributs — cliquez pour basculer 0 ↔ 1 :
      </p>
      <div id="clAdminAttrGrid" class="cl-admin-attr-grid"></div>

      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-top:20px;flex-wrap:wrap;gap:10px;">
        <button onclick="clAdminAnnuler()"
                style="background:white;color:#888;border:2px solid #ccc;border-radius:10px;
                       padding:10px 20px;font-size:.95em;font-weight:600;cursor:pointer;">
          🔄 Annuler les modifications
        </button>
        <button class="btn-success" onclick="clAdminSauvegarder()"
                style="padding:10px 24px;">
          💾 Enregistrer les modifications
        </button>
      </div>
      <div id="clAdminSaveMsg" style="margin-top:12px;display:none;"></div>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════
     CONTENEUR MODE ANIMAUX
══════════════════════════════════════════ -->
<div id="animauxMode">

<!-- ══════════════════════════════════════
     ÉCRAN 1 : SÉLECTION DE L'ANIMAL
══════════════════════════════════════════ -->
<div id="selectionScreen" class="screen active">
  <div class="card" style="max-width:860px;margin:0 auto;">
    <h1 style="text-align:center">🎭 Qui est-ce ?</h1>
    <p class="subtitle" style="text-align:center">
      Pensez à un animal parmi ceux ci-dessous — Akinator va essayer de le deviner,<br>
      et vous devrez deviner le sien !
    </p>
    <div id="animauxGrid" class="animals-grid">
      <p style="color:#aaa">Chargement des animaux…</p>
    </div>
    <div style="text-align:center;margin-top:10px">
      <button id="btnJaiChoisi" class="btn-primary" onclick="lancerJeu()">
        J'ai choisi ! 🚀
      </button>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════
     ÉCRAN 2 : JEU
══════════════════════════════════════════ -->
<div id="gameScreen" class="screen" style="max-width:860px;">
  <div id="turnBanner" class="turn-banner turn-joueur">🙋 C'est votre tour !</div>

  <div class="game-layout">

    <!-- Historique -->
    <div class="history-panel">
      <h3>📋 Historique</h3>
      <div id="historyList"></div>
    </div>

    <!-- Zone d'action -->
    <div class="action-panel">

      <!-- Tour du joueur ─────────────────── -->
      <div id="zoneJoueur">
        <h3>Posez une question à Akinator :</h3>
        <select id="questionSelect">
          <option value="">— Choisissez une question —</option>
        </select>
        <div class="row">
          <button class="btn-primary" onclick="poserQuestion()">❓ Poser la question</button>
          <button class="btn-danger" onclick="ouvrirModalProposition()">🏆 Proposer un animal</button>
        </div>
      </div>

      <!-- Réponse d'Akinator à la question du joueur ── -->
      <div id="zoneReponseAkinator" style="display:none">
        <h3>Akinator répond :</h3>
        <div class="answer-bubble" id="bulleReponse"></div>
        <p id="questionPoséeTexte" style="color:#888;font-size:.9em;margin-bottom:14px;"></p>
        <div class="row">
          <button class="btn-primary" onclick="tourAkinator()">➡️ Au tour d'Akinator</button>
          <button class="btn-danger" onclick="ouvrirModalProposition()">🏆 Proposer un animal</button>
        </div>
      </div>

      <!-- Tour d'Akinator : question ──────── -->
      <div id="zoneAkinatorQuestion" style="display:none">
        <h3>Akinator vous pose une question :</h3>
        <p id="akinatorQuestionTexte"
           style="font-size:1.15em;font-weight:600;color:#333;margin:14px 0;padding:14px;
                  background:var(--bg);border-radius:10px;"></p>
        <div id="akinatorAnswerBtns"></div>
      </div>

      <!-- Tour d'Akinator : proposition ───── -->
      <div id="zoneAkinatorProposition" style="display:none">
        <div class="proposal-box">
          <div style="font-size:1em">🤔 Akinator pense que votre animal est…</div>
          <div class="proposal-animal" id="akinatorProposalAnimal"></div>
          <div class="proposal-proba" id="akinatorProposalProba"></div>
        </div>
        <p style="color:#666;font-size:.95em;margin-bottom:14px;text-align:center">
          Est-ce bien votre animal ?
        </p>
        <div class="row" style="justify-content:center">
          <button class="btn-success" onclick="confirmerPropositionAkinator(true)">
            ✅ Oui, c'est lui !
          </button>
          <button class="btn-danger" onclick="confirmerPropositionAkinator(false)">
            ❌ Non, ce n'est pas lui
          </button>
        </div>
      </div>

      <!-- Révéler l'animal du joueur après erreur d'Akinator ── -->
      <div id="zoneRevelerAnimal" style="display:none">
        <h3>Quel était votre animal ?</h3>
        <p style="color:#888;font-size:.9em;margin-bottom:12px;">
          Sélectionnez votre animal pour qu'Akinator puisse apprendre !
        </p>
        <div id="revealGrid" class="animals-grid" style="max-height:280px;overflow-y:auto;"></div>
        <div style="text-align:center;margin-top:12px">
          <button id="btnConfirmerReveal" class="btn-primary" onclick="confirmerRevealAnimal()" disabled>
            ✔️ Confirmer
          </button>
        </div>
      </div>

    </div>
  </div>
</div>

<!-- ══════════════════════════════════════
     ÉCRAN 3 : FIN DE PARTIE
══════════════════════════════════════════ -->
<div id="endScreen" class="screen">
  <div class="card end-box">
    <div class="end-title" id="endTitle"></div>
    <div id="endMessage" style="color:#666;margin-bottom:20px;font-size:1.05em;"></div>
    <div class="reveal-row">
      <div class="reveal-card">
        <div class="label">Votre animal</div>
        <div class="big-emoji" id="emojiJoueur"></div>
        <div class="name" id="nomJoueur"></div>
      </div>
      <div class="reveal-card">
        <div class="label">Animal d'Akinator</div>
        <div class="big-emoji" id="emojiAkinator"></div>
        <div class="name" id="nomAkinator"></div>
      </div>
    </div>
    <button class="btn-primary" onclick="location.reload()">🔄 Rejouer</button>
  </div>
</div>

<!-- ══════════════════════════════════════
     MODAL : PROPOSER UN ANIMAL
══════════════════════════════════════════ -->
<div class="modal-overlay" id="modalProposition">
  <div class="modal-box">
    <h3>🏆 Quel est l'animal d'Akinator ?</h3>
    <div class="modal-grid" id="modalAnimauxGrid"></div>
    <div class="modal-actions">
      <button class="btn-outline" onclick="fermerModalProposition()">Annuler</button>
      <button class="btn-danger" id="btnConfirmerProp" onclick="confirmerProposition()" disabled>
        Confirmer
      </button>
    </div>
  </div>
</div>

<!-- ── Spinner ────────────────────────────── -->
<div class="loading-overlay" id="loadingOverlay">
  <div class="spinner"></div>
  <p style="color:var(--primary);font-weight:600">Chargement…</p>
</div>

</div><!-- /animauxMode -->


<!-- ══════════════════════════════════════
     CONTENEUR MODE CLASSIQUE
══════════════════════════════════════════ -->
<div id="classiqueMode" style="display:none">
<style>
  #classiqueMode { --bg: #f0fff8; --primary: #2a9d6e; --secondary: #1e7a55; }
  #classiqueMode .card { box-shadow: 0 6px 28px rgba(42,157,110,.13); }
  #classiqueMode .action-panel { box-shadow: 0 4px 18px rgba(42,157,110,.1); }
  #classiqueMode select:focus { border-color: #2a9d6e; }
  #classiqueMode h2, #classiqueMode h3 { color: #1e7a55; }
</style>

<!-- ══════════════════════════════════════
     CL-ÉCRAN 1 : SÉLECTION DU PERSONNAGE
══════════════════════════════════════════ -->
<div id="clSelectionScreen" class="screen active" style="max-width:600px;margin:0 auto;">
  <div class="card" style="text-align:center;">
    <h1>🧩 Qui est-ce ? <span style="font-size:.6em;color:var(--green)">— Version Classique</span></h1>
    <p class="subtitle">
      Choisissez secrètement un personnage sur votre plateau physique.<br>
      Akinator va essayer de le deviner, et vous devrez deviner le sien !
    </p>
    <div style="background:var(--bg);border-radius:14px;padding:24px 32px;margin:24px 0;display:inline-block;">
      <p style="font-size:1.15em;font-weight:600;color:var(--secondary);margin-bottom:6px;">
        🎭 Avez-vous choisi votre personnage ?
      </p>
      <p style="color:#888;font-size:.9em;">Ne le révélez pas — gardez-le secret !</p>
    </div>
    <br>
    <button class="btn-success" onclick="clLancerJeu()"
            style="font-size:1.1em;padding:16px 40px;">
      ✅ Oui, j'ai choisi ! Commencer
    </button>
  </div>
</div>

<!-- ══════════════════════════════════════
     CL-ÉCRAN 2 : JEU CLASSIQUE
══════════════════════════════════════════ -->
<div id="clGameScreen" class="screen" style="max-width:700px;margin:0 auto;">
  <div id="clTurnBanner" class="turn-banner turn-joueur">🙋 C'est votre tour !</div>
  <div class="action-panel">

      <!-- Tour du joueur ───────────────── -->
      <div id="clZoneJoueur">
        <!-- Compteur de personnages restants -->
        <div id="clRemainingBanner" style="background:linear-gradient(135deg,#e0fff4,#c8f5e4);
             border:2px solid #43b89c;border-radius:10px;padding:10px 16px;margin-bottom:14px;
             font-size:.95em;font-weight:600;color:#1e7a55;text-align:center;">
          🎭 Akinator a encore le choix entre <span id="clRemainingCount" style="font-size:1.2em">—</span> personnages
        </div>
        <h3>Posez une question à Akinator :</h3>
        <p style="color:#888;font-size:.85em;margin-bottom:10px">
          Cochez un ou plusieurs attributs — Akinator répond Oui si son personnage en possède <strong>au moins un</strong>.
        </p>
        <div id="clAttrList" class="cl-attr-list"></div>

        <!-- Prévisualisation -->
        <div class="cl-question-preview" id="clQuestionPreview">
          ← Cochez des attributs pour former votre question
        </div>

        <div class="row">
          <button class="btn-primary" id="clBtnPoser" onclick="clPoserQuestion()" disabled>
            ❓ Poser la question
          </button>
        </div>
      </div>

      <!-- Réponse d'Akinator ──────────── -->
      <div id="clZoneReponse" style="display:none">
        <h3>Akinator répond :</h3>
        <div id="clBulleReponse"></div>
        <p id="clQuestionPoseeTexte" style="color:#888;font-size:.88em;margin-bottom:14px;font-style:italic;"></p>
        <p id="clEliminationInfo" style="color:#43b89c;font-size:.88em;margin-bottom:14px;font-weight:600;"></p>
        <div class="row">
          <button class="btn-primary" onclick="clTourAkinator()">➡️ Au tour d'Akinator</button>
          <button class="btn-danger" onclick="clOuvrirModalProposition()">🏆 Proposer un personnage</button>
        </div>
      </div>

      <!-- Tour d'Akinator : question ───── -->
      <div id="clZoneAkinatorQuestion" style="display:none">
        <h3>Akinator vous pose une question :</h3>
        <p id="clAkinatorQuestionTexte"
           style="font-size:1.1em;font-weight:600;color:#333;margin:14px 0;padding:14px;
                  background:var(--bg);border-radius:10px;"></p>
        <p style="color:#888;font-size:.88em;margin-bottom:10px">
          Répondez <strong>Oui</strong> si votre personnage possède au moins un des attributs cités.
        </p>
        <div class="row" style="justify-content:center;gap:16px;">
          <button class="btn-oui" onclick="clRepondreAkinator(1)" style="min-width:130px;">
            ✅ Oui
          </button>
          <button class="btn-non" onclick="clRepondreAkinator(0)" style="min-width:130px;">
            ❌ Non
          </button>
        </div>
      </div>

      <!-- Tour d'Akinator : proposition ── -->
      <div id="clZoneAkinatorProposition" style="display:none">
        <div class="proposal-box">
          <div style="font-size:1em">🤔 Akinator pense que votre personnage est…</div>
          <div class="proposal-animal" id="clAkinatorProposalPerso"></div>
        </div>
        <p style="color:#666;font-size:.95em;margin-bottom:14px;text-align:center">
          Est-ce bien votre personnage ?
        </p>
        <div class="row" style="justify-content:center">
          <button class="btn-success" onclick="clConfirmerPropositionAkinator(true)">
            ✅ Oui, c'est lui !
          </button>
          <button class="btn-danger" onclick="clConfirmerPropositionAkinator(false)">
            ❌ Non, ce n'est pas lui
          </button>
        </div>
      </div>

    </div><!-- /action-panel -->
</div><!-- /clGameScreen -->

<!-- ══════════════════════════════════════
     CL-ÉCRAN 3 : FIN DE PARTIE
══════════════════════════════════════════ -->
<div id="clEndScreen" class="screen">
  <div class="card end-box">
    <div class="end-title" id="clEndTitle"></div>
    <div id="clEndMessage" style="color:#666;margin-bottom:20px;font-size:1.05em;"></div>
    <div class="reveal-row">
      <div class="reveal-card">
        <div class="label">Personnage d'Akinator</div>
        <div class="name" id="clNomAkinator" style="font-size:1.5em;font-weight:800;margin-top:6px;"></div>
      </div>
    </div>
    <div class="row" style="justify-content:center;gap:12px;">
      <button class="btn-success" onclick="clRejouer()">🔄 Rejouer (Classique)</button>
      <button class="btn-outline" onclick="switchMode()">🐾 Retour version Animaux</button>
    </div>
  </div>
</div>

</div><!-- /classiqueMode -->

<!-- Spinner ────────────────────────────── -->
<div class="loading-overlay" id="clLoadingOverlay">
  <div class="spinner" style="border-top-color:#43b89c"></div>
  <p style="color:#1e7a55;font-weight:600">Chargement…</p>
</div>

<!-- Modal proposition classique ── -->
<div class="modal-overlay" id="clModalProposition">
  <div class="modal-box" style="max-width:560px;">
    <h3 style="color:#1e7a55;">🏆 Quel est le personnage d'Akinator ?</h3>
    <p style="color:#888;font-size:.88em;margin-bottom:12px;">Cliquez sur un prénom pour le sélectionner.</p>
    <div id="clModalNomGrid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:6px;margin-bottom:16px;max-height:340px;overflow-y:auto;"></div>
    <div class="modal-actions">
      <button class="btn-outline" onclick="clFermerModalProposition()" style="color:#1e7a55;border-color:#1e7a55;">Annuler</button>
      <button class="btn-success" id="clBtnConfirmerProp" onclick="clConfirmerProposition()" disabled>
        ✅ Confirmer
      </button>
    </div>
  </div>
</div>

<script>
// ══════════════════════════════════════════
//  État global
// ══════════════════════════════════════════
let sessionId      = null;
let animaux        = [];
let questions      = [];
let animalJoueurIdx = null;
let animalPropose  = null;   // index sélectionné dans la modal

// Mapping emoji (noms courants en français)
const EMOJI_MAP = {
  'chien':['🐕','chien','chiens'], 'chat':['🐈','chat','chats'],
  'cheval':['🐴'], 'vache':['🐄'], 'cochon':['🐷','porc','porcin'],
  'mouton':['🐑'], 'chèvre':['🐐','chevre'], 'lion':['🦁'],
  'tigre':['🐯'], 'léopard':['🐆','leopard'], 'guépard':['🐆','guepard'],
  'éléphant':['🐘','elephant'], 'girafe':['🦒'], 'zèbre':['🦓','zebre'],
  'rhinocéros':['🦏','rhinoceros'], 'hippopotame':['🦛'],
  'singe':['🐒'], 'gorille':['🦍'], 'orang-outan':['🦧'],
  'ours':['🐻'], 'panda':['🐼'], 'ours polaire':['🐻‍❄️','ours blanc'],
  'renard':['🦊'], 'loup':['🐺'], 'cerf':['🦌'],
  'lapin':['🐰','lièvre'], 'écureuil':['🐿','ecureuil'],
  'souris':['🐭'], 'rat':['🐀'], 'hérisson':['🦔','herisson'],
  'chauve-souris':['🦇'], 'dauphin':['🐬'], 'baleine':['🐋'],
  'requin':['🦈'], 'pieuvre':['🐙'], 'crabe':['🦀'],
  'tortue':['🐢'], 'crocodile':['🐊'], 'serpent':['🐍'],
  'grenouille':['🐸'], 'pingouin':['🐧'], 'aigle':['🦅'],
  'hibou':['🦉'], 'flamant':['🦩'], 'perroquet':['🦜'],
  'poule':['🐔','poulet'], 'canard':['🦆'], 'cygne':['🦢'],
  'autruche':['🦤'], 'paon':['🦚'], 'pigeon':['🕊️','colombe'], 'papillon':['🦋'],
  'oie':['🦆'], 'koala':['🐨'], 'lézard':['🦎','lezard'],
  'abeille':['🐝'], 'coccinelle':['🐞'], 'araignée':['🕷'],
  'scorpion':['🦂'], 'escargot':['🐌'], 'poisson':['🐟'],
  'cheval marin':['🦑'], 'calamar':['🦑'],
};

function getEmoji(name) {
  const n = name.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g,"");
  for (const [key, vals] of Object.entries(EMOJI_MAP)) {
    const keyNorm = key.normalize("NFD").replace(/[\u0300-\u036f]/g,"");
    if (n.includes(keyNorm)) return vals[0];
    for (let i=1; i<vals.length; i++) {
      const v = vals[i].normalize("NFD").replace(/[\u0300-\u036f]/g,"");
      if (n.includes(v)) return vals[0];
    }
  }
  return '🐾';
}

// ══════════════════════════════════════════
//  Utilitaires d'affichage
// ══════════════════════════════════════════
// Gère uniquement les écrans du mode animaux
const ANIMAUX_SCREENS = ['selectionScreen','gameScreen','endScreen'];
function showScreen(id) {
  ANIMAUX_SCREENS.forEach(s => {
    const el = document.getElementById(s);
    if (el) el.classList.remove('active');
  });
  document.getElementById(id).classList.add('active');
}
function loading(on) {
  document.getElementById('loadingOverlay').classList.toggle('active', on);
}
function showZone(...ids) {
  ['zoneJoueur','zoneReponseAkinator','zoneAkinatorQuestion','zoneAkinatorProposition','zoneRevelerAnimal']
    .forEach(z => document.getElementById(z).style.display = ids.includes(z) ? 'block' : 'none');
}
function setBanner(text, type) {
  const b = document.getElementById('turnBanner');
  b.textContent = text;
  b.className = 'turn-banner ' + (type === 'joueur' ? 'turn-joueur' : 'turn-akinator');
}
function ajouterHistorique(question, reponse, parAkinator) {
  const div = document.createElement('div');
  div.className = 'history-item' + (parAkinator ? ' akinator-asks' : '');
  div.innerHTML = `<div class="history-q">${parAkinator ? '🤖' : '🙋'} ${question}</div>`
                + `<div class="history-a ${parAkinator ? 'akinator-ans' : ''}">→ ${reponse}</div>`;
  document.getElementById('historyList').prepend(div);
}

// ══════════════════════════════════════════
//  Écran 1 : Sélection
// ══════════════════════════════════════════
async function chargerAnimaux() {
  loading(true);
  try {
    // Timeout de 15s pour détecter les blocages réseau
    const controller = new AbortController();
    const timeoutId  = setTimeout(() => controller.abort(), 15000);

    const r = await fetch('/animals', { signal: controller.signal });
    clearTimeout(timeoutId);

    if (!r.ok) {
      const err = await r.json().catch(() => ({detail: `HTTP ${r.status}`}));
      throw new Error(err.detail || `Erreur serveur ${r.status}`);
    }

    const data = await r.json();
    if (!data.animaux || !data.questions) {
      throw new Error('Réponse du serveur incomplète (animaux ou questions manquants)');
    }

    animaux   = data.animaux;
    questions = data.questions;

    // Grille de sélection
    const grid = document.getElementById('animauxGrid');
    grid.innerHTML = '';
    animaux.forEach((nom, i) => {
      const card = document.createElement('div');
      card.className = 'animal-card';
      card.id = `sel-${i}`;
      card.innerHTML = `<span class="emoji">${getEmoji(nom)}</span>${nom}`;
      card.onclick = () => selectionnerAnimal(i);
      grid.appendChild(card);
    });

    // Remplir le dropdown des questions (pour le jeu)
    const sel = document.getElementById('questionSelect');
    sel.innerHTML = '<option value="">— Choisissez une question —</option>';
    questions.forEach((q, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = q;
      sel.appendChild(opt);
    });

  } catch(e) {
    const msg = e.name === 'AbortError'
      ? 'Délai dépassé (15s) — vérifiez que le serveur est démarré et que GOOGLE_CREDENTIALS est définie.'
      : `Erreur chargement : ${e.message}`;
    document.getElementById('animauxGrid').innerHTML =
      `<p style="color:#c62828;font-weight:600;grid-column:1/-1">⚠️ ${msg}</p>`;
    console.error('chargerAnimaux:', e);
  }
  loading(false);
}

function selectionnerAnimal(index) {
  // Grille affichée pour référence uniquement — pas de sélection requise
}

async function lancerJeu() {
  loading(true);
  try {
    const r = await fetch('/start_qui_est_ce', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({animal_joueur_index: -1})
    });
    const data = await r.json();
    sessionId = data.session_id;

    // Initialiser la modal de proposition (toujours les 24 animaux)
    const mg = document.getElementById('modalAnimauxGrid');
    mg.innerHTML = '';
    animaux.forEach((nom, i) => {
      const card = document.createElement('div');
      card.className = 'modal-animal';
      card.id = `prop-${i}`;
      card.innerHTML = `<span class="emoji">${getEmoji(nom)}</span>${nom}`;
      card.onclick = () => selectionnerProposition(i);
      mg.appendChild(card);
    });

    showScreen('gameScreen');
    showZone('zoneJoueur');
    setBanner('🙋 Votre tour — posez une question à Akinator !', 'joueur');
  } catch(e) {
    alert('Erreur démarrage : ' + e);
  }
  loading(false);
}

// ══════════════════════════════════════════
//  Tour du joueur : poser une question
// ══════════════════════════════════════════
async function poserQuestion() {
  const qIndex = parseInt(document.getElementById('questionSelect').value);
  if (isNaN(qIndex)) { alert('Veuillez choisir une question.'); return; }

  loading(true);
  try {
    const r = await fetch('/player_asks', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, question_index: qIndex})
    });
    const data = await r.json();

    // Afficher la réponse d'Akinator
    document.getElementById('bulleReponse').textContent = data.akinator_reponse;
    document.getElementById('questionPoséeTexte').textContent = '« ' + data.question_posee + ' »';
    ajouterHistorique(data.question_posee, data.akinator_reponse, false);

    // Réinitialiser le select
    document.getElementById('questionSelect').value = '';

    showZone('zoneReponseAkinator');
    setBanner('🤖 Akinator a répondu — à son tour maintenant !', 'akinator');
  } catch(e) {
    alert('Erreur : ' + e);
  }
  loading(false);
}

// ══════════════════════════════════════════
//  Tour d'Akinator
// ══════════════════════════════════════════
async function tourAkinator() {
  loading(true);
  try {
    const r    = await fetch(`/akinator_turn/${sessionId}`);
    const data = await r.json();

    if (data.type === 'question') {
      // Akinator pose une question
      document.getElementById('akinatorQuestionTexte').textContent = data.question_text;
      const btns = document.getElementById('akinatorAnswerBtns');
      btns.innerHTML = '';
      ['Oui','Plutôt oui','Je ne sais pas','Plutôt non','Non'].forEach((rep, i) => {
        const b = document.createElement('button');
        b.className = 'btn-answer';
        b.textContent = rep;
        b.onclick = () => repondreAkinator(i, data.question_text, rep);
        btns.appendChild(b);
      });
      showZone('zoneAkinatorQuestion');
      setBanner('🤖 Akinator vous pose une question !', 'akinator');

    } else {
      // Akinator propose un animal
      document.getElementById('akinatorProposalAnimal').textContent =
        getEmoji(data.animal) + ' ' + data.animal;
      document.getElementById('akinatorProposalProba').textContent =
        `Confiance : ${(data.probabilite * 100).toFixed(1)} %`;
      showZone('zoneAkinatorProposition');
      setBanner('🤔 Akinator fait une proposition !', 'akinator');
    }
  } catch(e) {
    alert('Erreur tour Akinator : ' + e);
  }
  loading(false);
}

// Le joueur répond à la question d'Akinator
async function repondreAkinator(reponseIndex, questionTexte, reponseTexte) {
  loading(true);
  try {
    const r = await fetch('/player_answers_akinator', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, reponse: reponseIndex})
    });
    const data = await r.json();
    // On n'ajoute PAS les questions d'Akinator dans l'historique

    // Si Akinator est déjà assez confiant après cette réponse → il propose directement
    if (data.proposition) {
      document.getElementById('akinatorProposalAnimal').textContent =
        getEmoji(data.proposition.animal) + ' ' + data.proposition.animal;
      document.getElementById('akinatorProposalProba').textContent =
        `Confiance : ${(data.proposition.probabilite * 100).toFixed(1)} %`;
      showZone('zoneAkinatorProposition');
      setBanner('🤔 Akinator fait une proposition !', 'akinator');
    } else {
      // Retour au tour du joueur
      showZone('zoneJoueur');
      setBanner('🙋 Votre tour — posez une question à Akinator !', 'joueur');
    }
  } catch(e) {
    alert('Erreur : ' + e);
  }
  loading(false);
}

// Réponse du joueur à la proposition d'Akinator (Oui / Non)
async function confirmerPropositionAkinator(correct) {
  loading(true);
  try {
    const r    = await fetch('/confirm_akinator_proposal', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, correct})
    });
    const data = await r.json();

    if (data.correct) {
      // Akinator avait raison → fin de partie
      afficherFin(
        '😮 Akinator a trouvé ! Il gagne !',
        `Akinator a correctement deviné votre animal.`,
        data.animal_joueur,
        data.animal_akinator
      );
    } else {
      // Akinator s'est trompé → demander l'animal du joueur pour l'entraînement
      const grid = document.getElementById('revealGrid');
      grid.innerHTML = '';
      animaux.forEach((nom, i) => {
        const card = document.createElement('div');
        card.className = 'animal-card';
        card.id = `rev-${i}`;
        card.innerHTML = `<span class="emoji">${getEmoji(nom)}</span>${nom}`;
        card.onclick = () => {
          document.querySelectorAll('#revealGrid .animal-card').forEach(c => c.classList.remove('selected'));
          card.classList.add('selected');
          revealAnimalIdx = i;
          document.getElementById('btnConfirmerReveal').disabled = false;
        };
        grid.appendChild(card);
      });
      showZone('zoneRevelerAnimal');
      setBanner('🎉 Bien joué ! Dites-nous quel était votre animal', 'joueur');
    }
  } catch(e) {
    alert('Erreur : ' + e);
  }
  loading(false);
}

// Le joueur déclare son animal (pour apprentissage) — utilisé dans tous les cas post-partie
let revealAnimalIdx = null;
async function confirmerRevealAnimal() {
  if (revealAnimalIdx === null) return;
  loading(true);
  try {
    const r = await fetch('/declare_joueur_animal', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, animal_index: revealAnimalIdx})
    });
    const data = await r.json();
    afficherFin(
      data.titre || '🏆 Partie terminée !',
      data.message || '',
      data.animal_joueur,
      data.animal_akinator
    );
  } catch(e) {
    alert('Erreur : ' + e);
  }
  loading(false);
}

// ══════════════════════════════════════════
//  Modal : proposition du joueur
// ══════════════════════════════════════════
function ouvrirModalProposition() {
  animalPropose = null;
  document.querySelectorAll('.modal-animal').forEach(c => c.classList.remove('selected'));
  document.getElementById('btnConfirmerProp').disabled = true;
  document.getElementById('modalProposition').classList.add('active');
}
function fermerModalProposition() {
  document.getElementById('modalProposition').classList.remove('active');
}
function selectionnerProposition(index) {
  document.querySelectorAll('.modal-animal').forEach(c => c.classList.remove('selected'));
  document.getElementById(`prop-${index}`).classList.add('selected');
  animalPropose = index;
  document.getElementById('btnConfirmerProp').disabled = false;
}

async function confirmerProposition() {
  if (animalPropose === null) return;
  fermerModalProposition();
  loading(true);
  try {
    const r = await fetch('/player_proposes', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: sessionId, animal_index: animalPropose})
    });
    const data = await r.json();

    if (data.correct) {
      // ✅ Joueur a trouvé l'animal d'Akinator
      // → demander quel était son propre animal pour l'apprentissage
      afficherRevealPourApprentissage(
        '🏆 Bravo ! Vous avez trouvé !',
        `Vous avez deviné l'animal d'Akinator : ${data.animal_akinator}.<br>Pour qu'Akinator apprenne, dites-lui quel était votre animal :`,
        data.animal_akinator
      );
    } else {
      // ❌ Joueur s'est trompé → Akinator joue en solo
      setBanner('🤖 Vous vous êtes trompé… Akinator joue seul !', 'akinator');
      showZone('zoneAkinatorQuestion'); // zone neutre le temps du chargement
      document.getElementById('akinatorQuestionTexte').textContent = 'Akinator réfléchit…';
      document.getElementById('akinatorAnswerBtns').innerHTML = '';
      await akinatorSoloPropose(data.animal_akinator_nom);
    }
  } catch(e) {
    alert('Erreur : ' + e);
  }
  loading(false);
}

// Akinator propose son meilleur candidat après que le joueur s'est trompé
let _soloAnimal = '';
let _soloAkinatorRevele = '';

async function akinatorSoloPropose(nomAkinatorRevele) {
  loading(true);
  try {
    const r    = await fetch(`/akinator_solo_propose/${sessionId}`);
    const data = await r.json();

    // Stocker dans des variables pour éviter les problèmes d'apostrophes dans onclick
    _soloAnimal        = data.animal;
    _soloAkinatorRevele = nomAkinatorRevele || '';

    // Afficher la proposition solo d'Akinator
    document.getElementById('akinatorProposalAnimal').textContent =
      getEmoji(data.animal) + ' ' + data.animal;
    document.getElementById('akinatorProposalProba').textContent =
      `Confiance : ${(data.probabilite * 100).toFixed(1)} %`;
    showZone('zoneAkinatorProposition');

    // Remplacer les boutons Oui/Non par la logique "solo"
    const box = document.getElementById('zoneAkinatorProposition');
    box.querySelectorAll('.solo-btn').forEach(b => b.remove());

    const div = document.createElement('div');
    div.className = 'row solo-btn';
    div.style.justifyContent = 'center';
    div.style.marginTop = '14px';

    const btnOui = document.createElement('button');
    btnOui.className = 'btn-success solo-btn';
    btnOui.textContent = `✅ Oui, c'est mon animal !`;
    btnOui.onclick = () => resultatSolo(true);

    const btnNon = document.createElement('button');
    btnNon.className = 'btn-danger solo-btn';
    btnNon.textContent = `❌ Non, ce n'est pas lui`;
    btnNon.onclick = () => resultatSolo(false);

    div.appendChild(btnOui);
    div.appendChild(btnNon);
    box.appendChild(div);
    setBanner('🤔 Akinator fait une dernière proposition !', 'akinator');
  } catch(e) {
    alert('Erreur solo : ' + e);
  }
  loading(false);
}

async function resultatSolo(correct) {
  const animalProposeNom  = _soloAnimal;
  const nomAkinatorRevele = _soloAkinatorRevele;

  // Supprimer les boutons solo pour éviter double-clic
  document.querySelectorAll('.solo-btn').forEach(b => b.remove());

  if (correct) {
    // ✅ Akinator a trouvé → on connaît l'animal du joueur = animal proposé
    // → apprentissage direct via declare_joueur_animal avec l'index connu
    loading(true);
    try {
      const r = await fetch('/akinator_solo_correct', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({session_id: sessionId})
      });
      const data = await r.json();
      afficherFin(
        '😮 Akinator a quand même trouvé !',
        `Votre animal était bien ${data.animal_joueur}. Akinator a appris !`,
        data.animal_joueur,
        data.animal_akinator
      );
    } catch(e) { alert('Erreur : ' + e); }
    loading(false);
  } else {
    // ❌ Akinator s'est trompé → on ne sait pas quel était l'animal du joueur
    // → demander au joueur pour apprendre
    afficherRevealPourApprentissage(
      `🤝 Ni vous ni Akinator n'avez trouvé !`,
      `Akinator n'a pas trouvé votre animal. Dites-le lui pour qu'il apprenne :`,
      nomAkinatorRevele
    );
  }
}

// Affiche la grille de révélation de l'animal du joueur pour l'apprentissage
// (utilisée après victoire du joueur ET après victoire d'Akinator en solo)
function afficherRevealPourApprentissage(titre, messageHtml, nomAkinatorRevele) {
  // Adapter le titre et le message de la zone
  const zone = document.getElementById('zoneRevelerAnimal');
  zone.querySelector('h3').textContent = titre;
  zone.querySelector('p').innerHTML    = messageHtml;

  const grid = document.getElementById('revealGrid');
  grid.innerHTML = '';
  animaux.forEach((nom, i) => {
    const card = document.createElement('div');
    card.className = 'animal-card';
    card.id = `rev-${i}`;
    card.innerHTML = `<span class="emoji">${getEmoji(nom)}</span>${nom}`;
    card.onclick = () => {
      document.querySelectorAll('#revealGrid .animal-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      revealAnimalIdx = i;
      document.getElementById('btnConfirmerReveal').disabled = false;
    };
    grid.appendChild(card);
  });
  showZone('zoneRevelerAnimal');
  setBanner('📚 Aidez Akinator à apprendre !', 'joueur');
}

// ══════════════════════════════════════════
//  Fin de partie
// ══════════════════════════════════════════
function afficherFin(titre, message, nomJoueur, nomAkinator) {
  document.getElementById('endTitle').textContent   = titre;
  document.getElementById('endMessage').textContent = message;
  const estInconnu = !nomJoueur || nomJoueur === '?';
  document.getElementById('emojiJoueur').textContent   = estInconnu ? '❓' : getEmoji(nomJoueur);
  document.getElementById('nomJoueur').textContent     = estInconnu ? '(non révélé)' : nomJoueur;
  document.getElementById('emojiAkinator').textContent = getEmoji(nomAkinator);
  document.getElementById('nomAkinator').textContent   = nomAkinator;
  showScreen('endScreen');
}

// ══════════════════════════════════════════
//  Panneau technique (admin)
// ══════════════════════════════════════════
let adminLoaded        = false;
let adminData          = null;   // { animaux, questions, donnees }
let selectedQIndex     = null;   // index de la question en cours d'édition

// Convertit une valeur numérique vers l'option la plus proche du <select>
const VALEURS_OPTIONS  = ['1', '0.75', '0.5', '0.25', '0'];
function valeurVersOption(v) {
  let best = '0.5', bestDist = Infinity;
  VALEURS_OPTIONS.forEach(opt => {
    const d = Math.abs(parseFloat(opt) - parseFloat(v));
    if (d < bestDist) { bestDist = d; best = opt; }
  });
  return best;
}

function afficherMsg(elId, texte, succes) {
  const el = document.getElementById(elId);
  el.style.cssText = `display:block;padding:12px 16px;border-radius:10px;margin-top:12px;font-weight:600;`
    + (succes
      ? 'background:#e8f8f0;color:#2e7d52;'
      : 'background:#fdecea;color:#c62828;');
  el.textContent = texte;
  setTimeout(() => { el.style.display = 'none'; }, 4000);
}

async function toggleAdmin() {
  if (clCurrentMode) { toggleAdminClassique(); return; }
  const panel = document.getElementById('adminPanel');
  const isOpen = panel.classList.toggle('active');
  if (isOpen && !adminLoaded) { await chargerAdmin(); adminLoaded = true; }
}

function toggleAdminClassique() {
  const panel = document.getElementById('adminClassiquePanel');
  const isOpen = panel.classList.toggle('active');
  if (isOpen) clAdminCharger();
}

// ══════════════════════════════════════════════════════════════
//  PANNEAU ADMIN CLASSIQUE
// ══════════════════════════════════════════════════════════════
var clAdminData        = null;
var clAdminPersoSel    = null;
var clAdminValeursCopy = [];

async function clAdminCharger() {
  try {
    const r = await fetch('/classique/admin/data');
    if (!r.ok) throw new Error('Erreur ' + r.status);
    clAdminData = await r.json();
    clAdminBuildPersoGrid();
    document.getElementById('clAdminEditSection').style.display = 'none';
    clAdminPersoSel = null;
  } catch(e) {
    document.getElementById('clAdminPersoGrid').innerHTML =
      '<p style="color:red">Erreur chargement : ' + e.message + '</p>';
  }
}

function clAdminBuildPersoGrid() {
  const grid = document.getElementById('clAdminPersoGrid');
  grid.innerHTML = '';
  clAdminData.personnages.forEach((nom, i) => {
    const card = document.createElement('div');
    card.className = 'cl-admin-perso-card';
    card.id = 'clAdminPC-' + i;
    card.textContent = nom;
    card.onclick = () => clAdminSelectionnerPersonnage(i);
    grid.appendChild(card);
  });
}

function clAdminSelectionnerPersonnage(idx) {
  document.querySelectorAll('.cl-admin-perso-card')
          .forEach(c => c.classList.remove('cl-admin-selected'));
  document.getElementById('clAdminPC-' + idx).classList.add('cl-admin-selected');
  clAdminPersoSel    = idx;
  clAdminValeursCopy = clAdminData.donnees.map(row => row[idx]);
  document.getElementById('clAdminEditBadge').textContent = clAdminData.personnages[idx];
  document.getElementById('clAdminNomInput').value        = clAdminData.personnages[idx];
  clAdminBuildAttrGrid();
  const sec = document.getElementById('clAdminEditSection');
  sec.style.display = 'block';
  sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
  document.getElementById('clAdminSaveMsg').style.display = 'none';
}

function clAdminBuildAttrGrid() {
  const grid = document.getElementById('clAdminAttrGrid');
  grid.innerHTML = '';
  clAdminData.attributs.forEach((attr, i) => {
    const val = clAdminValeursCopy[i];
    const div = document.createElement('div');
    div.className = 'cl-admin-attr-item val-' + val;
    div.id = 'clAdminAttr-' + i;
    div.innerHTML = '<span class="cl-admin-attr-badge">' + val + '</span><span>' + attr + '</span>';
    div.onclick = () => clAdminToggleAttr(i);
    grid.appendChild(div);
  });
}

function clAdminToggleAttr(i) {
  clAdminValeursCopy[i] = clAdminValeursCopy[i] === 1 ? 0 : 1;
  const val = clAdminValeursCopy[i];
  const div = document.getElementById('clAdminAttr-' + i);
  div.className = 'cl-admin-attr-item val-' + val;
  div.querySelector('.cl-admin-attr-badge').textContent = val;
}

function clAdminAnnuler() {
  if (clAdminPersoSel === null) return;
  clAdminValeursCopy = clAdminData.donnees.map(row => row[clAdminPersoSel]);
  document.getElementById('clAdminNomInput').value = clAdminData.personnages[clAdminPersoSel];
  clAdminBuildAttrGrid();
  document.getElementById('clAdminSaveMsg').style.display = 'none';
}

async function clAdminSauvegarder() {
  if (clAdminPersoSel === null) return;
  const nouveauNom = document.getElementById('clAdminNomInput').value.trim();
  if (!nouveauNom) { alert('Le nom ne peut pas être vide.'); return; }

  const ancienNom = clAdminData.personnages[clAdminPersoSel];
  const changements = clAdminData.attributs
    .map((attr, i) => ({ attr, ancien: clAdminData.donnees[i][clAdminPersoSel], nouveau: clAdminValeursCopy[i] }))
    .filter(x => x.ancien !== x.nouveau);

  var NL = String.fromCharCode(10);
  var lignes = ['Confirmer les modifications pour ' + ancienNom + ' ?'];
  if (nouveauNom !== ancienNom) lignes.push('Nom : ' + ancienNom + ' -> ' + nouveauNom);
  changements.forEach(function(x) { lignes.push(x.attr + ' : ' + x.ancien + ' -> ' + x.nouveau); });
  if (changements.length === 0 && nouveauNom === ancienNom) {
    alert('Aucune modification detectee.'); return;
  }
  if (!confirm(lignes.join(NL))) return;

  try {
    const r = await fetch('/classique/admin/update_personnage', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: clAdminPersoSel, nom: nouveauNom, valeurs: clAdminValeursCopy })
    });
    if (!r.ok) throw new Error((await r.json()).detail);

    clAdminData.personnages[clAdminPersoSel] = nouveauNom;
    clAdminData.donnees.forEach((row, i) => { row[clAdminPersoSel] = clAdminValeursCopy[i]; });
    clAdminBuildPersoGrid();
    document.getElementById('clAdminPC-' + clAdminPersoSel).classList.add('cl-admin-selected');
    document.getElementById('clAdminEditBadge').textContent = nouveauNom;

    const msgEl = document.getElementById('clAdminSaveMsg');
    msgEl.style.cssText = 'display:block;padding:12px 16px;border-radius:10px;margin-top:12px;' +
      'font-weight:600;background:#e8f8f0;color:#2e7d52;';
    msgEl.textContent = '✅ ' + nouveauNom + ' mis à jour avec succès !';
    setTimeout(() => { msgEl.style.display = 'none'; }, 4000);

    if (clPersonnages && clPersonnages[clAdminPersoSel] !== undefined)
      clPersonnages[clAdminPersoSel] = nouveauNom;
  } catch(e) {
    const msgEl = document.getElementById('clAdminSaveMsg');
    msgEl.style.cssText = 'display:block;padding:12px 16px;border-radius:10px;margin-top:12px;' +
      'font-weight:600;background:#fdecea;color:#c62828;';
    msgEl.textContent = '❌ Erreur : ' + e.message;
    setTimeout(() => { msgEl.style.display = 'none'; }, 5000);
  }
}

async function chargerAdmin() {
  try {
    const r = await fetch('/admin/data');
    adminData = await r.json();  // { animaux, questions, donnees }

    // ── Tableau des questions (cliquable) ──
    rebuildTable();

    // ── Grille "Ajouter" ──
    buildAnswerGrid('animalAnswerGrid', null);

  } catch(e) {
    document.getElementById('adminQuestionsTable').innerHTML =
      `<p style="color:red">Erreur : ${e}</p>`;
  }
}

function rebuildTable() {
  const qs = adminData.questions;
  let html = `<table class="q-table">
    <thead><tr><th style="width:40px">#</th><th>Question</th></tr></thead><tbody>`;
  qs.forEach((q, i) => {
    const sel = (i === selectedQIndex) ? ' selected-row' : '';
    html += `<tr class="clickable${sel}" id="qrow-${i}" onclick="selectionnerQuestion(${i})">
               <td>${i + 1}</td><td>${q}</td></tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('adminQuestionsTable').innerHTML = html;
}

// Construit une grille de selects; valeurs = tableau float ou null (→ 0.5 par défaut)
function buildAnswerGrid(gridId, valeurs) {
  const grid = document.getElementById(gridId);
  grid.innerHTML = '';
  adminData.animaux.forEach((nom, i) => {
    const div = document.createElement('div');
    div.className = 'animal-answer-row';
    const val = valeurs ? valeurVersOption(valeurs[i]) : '0.5';
    div.innerHTML = `
      <div class="a-name">${getEmoji(nom)} ${nom}</div>
      <select id="${gridId}-${i}">
        <option value="1"   ${val==='1'    ? 'selected':''}>Oui</option>
        <option value="0.75"${val==='0.75' ? 'selected':''}>Plutôt oui</option>
        <option value="0.5" ${val==='0.5'  ? 'selected':''}>Je ne sais pas</option>
        <option value="0.25"${val==='0.25' ? 'selected':''}>Plutôt non</option>
        <option value="0"   ${val==='0'    ? 'selected':''}>Non</option>
      </select>`;
    grid.appendChild(div);
  });
}

function collectValeurs(gridId) {
  const vals = [];
  let i = 0;
  while (document.getElementById(`${gridId}-${i}`)) {
    vals.push(parseFloat(document.getElementById(`${gridId}-${i}`).value));
    i++;
  }
  return vals;
}

// ── Sélectionner une question existante pour édition ──
function selectionnerQuestion(index) {
  const sec = document.getElementById('editSection');

  // Si on reclique la même ligne → fermer et désélectionner
  if (selectedQIndex === index) {
    selectedQIndex = null;
    rebuildTable();
    sec.style.display = 'none';
    return;
  }

  selectedQIndex = index;
  rebuildTable();

  // Remplir le formulaire d'édition
  document.getElementById('editBadge').textContent = `#${index + 1}`;
  document.getElementById('editQuestionText').value = adminData.questions[index];
  buildAnswerGrid('editAnimalGrid', adminData.donnees[index]);

  // Afficher la section
  sec.style.display = 'block';
  sec.scrollIntoView({behavior:'smooth', block:'start'});
  document.getElementById('editSaveMsg').style.display = 'none';
}

// ── Annuler → remettre les valeurs du fichier ──
function annulerModifications() {
  if (selectedQIndex === null) return;
  document.getElementById('editQuestionText').value = adminData.questions[selectedQIndex];
  buildAnswerGrid('editAnimalGrid', adminData.donnees[selectedQIndex]);
  document.getElementById('editSaveMsg').style.display = 'none';
}

// ── Sauvegarder les modifications ──
async function sauvegarderModification() {
  if (selectedQIndex === null) return;
  const texte   = document.getElementById('editQuestionText').value.trim();
  if (!texte) { alert('Le texte de la question ne peut pas être vide.'); return; }
  const valeurs = collectValeurs('editAnimalGrid');

  loading(true);
  try {
    const r = await fetch('/admin/update_question', {
      method: 'PUT',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({index: selectedQIndex, question: texte, valeurs})
    });
    const data = await r.json();

    // Mettre à jour adminData localement
    adminData.questions[selectedQIndex] = texte;
    adminData.donnees[selectedQIndex]   = valeurs;
    rebuildTable();

    // Mettre à jour dropdown de jeu
    if (questions.length > selectedQIndex) {
      questions[selectedQIndex] = texte;
      const sel = document.getElementById('questionSelect');
      if (sel.options[selectedQIndex + 1])
        sel.options[selectedQIndex + 1].textContent = texte;
    }
    afficherMsg('editSaveMsg', `✅ Question #${selectedQIndex + 1} mise à jour !`, true);
  } catch(e) {
    afficherMsg('editSaveMsg', `❌ Erreur : ${e}`, false);
  }
  loading(false);
}

// ── Supprimer une question ──
async function supprimerQuestion() {
  if (selectedQIndex === null) return;
  const texte = adminData.questions[selectedQIndex];
  const ok = confirm(`Êtes-vous sûr de vouloir supprimer cette question ?\n\n"${texte}"\n\nCette action est irréversible.`);
  if (!ok) return;

  loading(true);
  try {
    const r = await fetch('/admin/delete_question', {
      method: 'DELETE',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({index: selectedQIndex})
    });
    await r.json();

    // Mettre à jour adminData localement
    adminData.questions.splice(selectedQIndex, 1);
    adminData.donnees.splice(selectedQIndex, 1);

    // Mettre à jour dropdown de jeu
    if (questions.length > selectedQIndex) {
      questions.splice(selectedQIndex, 1);
      const sel = document.getElementById('questionSelect');
      if (sel.options[selectedQIndex + 1])
        sel.options[selectedQIndex + 1].remove();
      // Recalculer les indices des options restantes
      for (let i = selectedQIndex; i < sel.options.length - 1; i++)
        sel.options[i + 1].value = i;
    }

    selectedQIndex = null;
    document.getElementById('editSection').style.display = 'none';
    rebuildTable();
    afficherMsg('adminSaveMsg', `✅ Question supprimée avec succès.`, true);
  } catch(e) {
    afficherMsg('editSaveMsg', `❌ Erreur : ${e}`, false);
  }
  loading(false);
}

async function sauvegarderNouvelleQuestion() {
  const texte = document.getElementById('newQuestionText').value.trim();
  if (!texte) { alert('Veuillez saisir le texte de la question.'); return; }
  const valeurs = collectValeurs('animalAnswerGrid');

  loading(true);
  try {
    const r = await fetch('/admin/add_question', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({question: texte, valeurs})
    });
    const data = await r.json();

    // Mettre à jour adminData localement
    adminData.questions.push(texte);
    adminData.donnees.push(valeurs);
    rebuildTable();

    // Vider le champ + reset selects
    document.getElementById('newQuestionText').value = '';
    buildAnswerGrid('animalAnswerGrid', null);

    // Mettre à jour dropdown de jeu
    if (questions.length > 0) {
      const sel = document.getElementById('questionSelect');
      const opt = document.createElement('option');
      opt.value   = data.question_index;
      opt.textContent = texte;
      sel.appendChild(opt);
      questions.push(texte);
    }
    afficherMsg('adminSaveMsg', `✅ Question "${texte}" ajoutée avec succès !`, true);
  } catch(e) {
    afficherMsg('adminSaveMsg', `❌ Erreur : ${e}`, false);
  }
  loading(false);
}

// ══════════════════════════════════════════
//  Initialisation
// ══════════════════════════════════════════
window.onload = chargerAnimaux;

// ══════════════════════════════════════════════════════════════
//  VERSION CLASSIQUE — état global & utilitaires
// ══════════════════════════════════════════════════════════════
var clCurrentMode = false;
var clSessionId   = null;
var clPersonnages = [];
var clAttributs   = [];
var clAttrsChecked = new Set();

function clLoading(on) {
  var el = document.getElementById('clLoadingOverlay');
  if (el) el.classList.toggle('active', on);
}
function clShowZone() {
  var ids = Array.from(arguments);
  ['clZoneJoueur','clZoneReponse','clZoneAkinatorQuestion','clZoneAkinatorProposition']
    .forEach(function(z) {
      var el = document.getElementById(z);
      if (el) el.style.display = ids.indexOf(z) >= 0 ? 'block' : 'none';
    });
}
function clSetBanner(text, type) {
  var b = document.getElementById('clTurnBanner');
  if (!b) return;
  b.textContent = text;
  b.className = 'turn-banner ' + (type === 'joueur' ? 'turn-joueur' : 'turn-akinator');
}

// Gère uniquement les écrans du mode classique
var CL_SCREENS = ['clSelectionScreen','clGameScreen','clEndScreen'];
function clShowScreen(id) {
  CL_SCREENS.forEach(function(s) {
    var el = document.getElementById(s);
    if (el) el.classList.remove('active');
  });
  var target = document.getElementById(id);
  if (target) target.classList.add('active');
}

// ══════════════════════════════════════════════════════════════
//  Bascule de mode (Animaux ↔ Classique)
// ══════════════════════════════════════════════════════════════
function switchMode() {
  clCurrentMode = !clCurrentMode;
  var btnCl = document.getElementById('btnClassique');
  if (clCurrentMode) {
    document.getElementById('animauxMode').style.display   = 'none';
    document.getElementById('classiqueMode').style.display = 'block';
    clShowScreen('clSelectionScreen');
    btnCl.textContent = '🐾 Version Animaux';
    btnCl.classList.add('active-mode');
    if (clPersonnages.length === 0) chargerPersonnagesClassique();
  } else {
    document.getElementById('animauxMode').style.display   = 'block';
    document.getElementById('classiqueMode').style.display = 'none';
    btnCl.textContent = '🧩 Version Classique';
    btnCl.classList.remove('active-mode');
  }
}


// ══════════════════════════════════════════════════════════════
//  Chargement des attributs
// ══════════════════════════════════════════════════════════════
async function chargerPersonnagesClassique() {
  clLoading(true);
  try {
    const r = await fetch('/classique/personnages');
    if (!r.ok) throw new Error('Erreur serveur ' + r.status);
    const data = await r.json();
    clPersonnages = data.personnages;
    clAttributs   = data.attributs;
  } catch(e) {
    alert('Erreur chargement : ' + e.message);
  }
  clLoading(false);
}


// ══════════════════════════════════════════════════════════════
//  Démarrage de la partie
// ══════════════════════════════════════════════════════════════
async function clLancerJeu() {
  clLoading(true);
  try {
    const r = await fetch('/classique/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({personnage_joueur_index: 0})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();
    clSessionId    = data.session_id;
    clPersonnages  = data.personnages;
    clAttributs    = data.attributs;
    clAttrsChecked = new Set();

    clBuildAttrList();
    clBuildModalNomGrid();
    clUpdateRemainingCount(data.personnages.length);
    clShowScreen('clGameScreen');
    clShowZone('clZoneJoueur');
    clSetBanner('🙋 Votre tour — posez une question à Akinator !', 'joueur');
  } catch(e) {
    alert('Erreur démarrage : ' + e.message);
  }
  clLoading(false);
}

function clRejouer() {
  clSessionId      = null;
  clAttrsChecked   = new Set();
  clPropositionIdx = null;
  clShowScreen('clSelectionScreen');
}

function clUpdateRemainingCount(n) {
  const el = document.getElementById('clRemainingCount');
  if (el) el.textContent = n;
}

// Grille de prénoms pour la modal (sans emoji)
var clPropositionIdx = null;
function clBuildModalNomGrid() {
  const grid = document.getElementById('clModalNomGrid');
  if (!grid) return;
  grid.innerHTML = '';
  clPersonnages.forEach((nom, i) => {
    const btn = document.createElement('button');
    btn.textContent = nom;
    btn.style.cssText = 'padding:8px 6px;border-radius:8px;border:2px solid #dde;background:#f8f8f8;' +
      'font-size:.9em;font-weight:600;cursor:pointer;color:#333;transition:all .15s;';
    btn.id = 'clPropBtn-' + i;
    btn.onclick = () => clSelectionnerProposition(i);
    grid.appendChild(btn);
  });
}

function clOuvrirModalProposition() {
  clPropositionIdx = null;
  document.querySelectorAll('#clModalNomGrid button').forEach(b => {
    b.style.background    = '#f8f8f8';
    b.style.borderColor   = '#dde';
    b.style.color         = '#333';
  });
  document.getElementById('clBtnConfirmerProp').disabled = true;
  document.getElementById('clModalProposition').classList.add('active');
}
function clFermerModalProposition() {
  document.getElementById('clModalProposition').classList.remove('active');
}
function clSelectionnerProposition(index) {
  document.querySelectorAll('#clModalNomGrid button').forEach(b => {
    b.style.background  = '#f8f8f8';
    b.style.borderColor = '#dde';
    b.style.color       = '#333';
  });
  const btn = document.getElementById('clPropBtn-' + index);
  if (btn) {
    btn.style.background  = '#d4f5e9';
    btn.style.borderColor = '#43b89c';
    btn.style.color       = '#1e7a55';
  }
  clPropositionIdx = index;
  document.getElementById('clBtnConfirmerProp').disabled = false;
}

async function clConfirmerProposition() {
  if (clPropositionIdx === null) return;
  clFermerModalProposition();
  clLoading(true);
  try {
    const r = await fetch('/classique/player_proposes', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: clSessionId, personnage_index: clPropositionIdx})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();
    if (data.correct) {
      clAfficherFin('🏆 Bravo ! Vous avez trouvé !',
                    "Vous avez deviné le personnage d'Akinator !",
                    data.personnage_akinator);
    } else {
      alert("❌ Ce n'est pas le bon personnage ! Continuez à poser des questions.");
    }
  } catch(e) {
    alert('Erreur : ' + e.message);
  }
  clLoading(false);
}


// ══════════════════════════════════════════════════════════════
//  Liste des attributs (re-cochables à chaque tour)
// ══════════════════════════════════════════════════════════════
function clBuildAttrList() {
  const list = document.getElementById('clAttrList');
  list.innerHTML = '';
  clAttributs.forEach((attr, i) => {
    const div = document.createElement('div');
    div.className = 'cl-attr-item';
    div.id = 'clAttr-' + i;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id   = 'clCb-' + i;
    cb.onchange = () => clToggleAttr(i, cb.checked);
    const lbl = document.createElement('label');
    lbl.htmlFor     = 'clCb-' + i;
    lbl.textContent = attr;
    lbl.style.cursor = 'pointer';
    lbl.style.flex   = '1';
    div.appendChild(cb);
    div.appendChild(lbl);
    div.onclick = (e) => {
      if (e.target !== cb && e.target !== lbl) { cb.checked = !cb.checked; clToggleAttr(i, cb.checked); }
    };
    list.appendChild(div);
  });
}

function clResetCheckboxes() {
  clAttrsChecked = new Set();
  clAttributs.forEach((_, i) => {
    const cb  = document.getElementById('clCb-' + i);
    const div = document.getElementById('clAttr-' + i);
    if (cb)  cb.checked = false;
    if (div) div.classList.remove('checked');
  });
  document.getElementById('clBtnPoser').disabled = true;
  clUpdateQuestionPreview();
}

function clToggleAttr(idx, checked) {
  const div = document.getElementById('clAttr-' + idx);
  if (checked) { clAttrsChecked.add(idx);    div.classList.add('checked');    }
  else         { clAttrsChecked.delete(idx); div.classList.remove('checked'); }
  clUpdateQuestionPreview();
  document.getElementById('clBtnPoser').disabled = clAttrsChecked.size === 0;
}

function clUpdateQuestionPreview() {
  const preview = document.getElementById('clQuestionPreview');
  if (clAttrsChecked.size === 0) {
    preview.textContent = '← Cochez des attributs pour former votre question';
    preview.style.color = '#aaa';
    return;
  }
  const textes = [...clAttrsChecked].map(i => clAttributs[i]);
  preview.textContent = '❓ Le personnage ' + textes.join(' ou ') + ' ?';
  preview.style.color = 'var(--secondary)';
}


// ══════════════════════════════════════════════════════════════
//  Tour du joueur
// ══════════════════════════════════════════════════════════════
async function clPoserQuestion() {
  if (clAttrsChecked.size === 0) { alert('Cochez au moins un attribut.'); return; }
  clLoading(true);
  try {
    const r = await fetch('/classique/player_asks', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: clSessionId, attr_indices: [...clAttrsChecked]})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();

    const bulle = document.getElementById('clBulleReponse');
    if (data.reponse === 'Oui') {
      bulle.className   = 'cl-answer-bubble-oui';
      bulle.textContent = '✅ Oui !';
    } else {
      bulle.className   = 'cl-answer-bubble-non';
      bulle.textContent = '❌ Non !';
    }
    document.getElementById('clQuestionPoseeTexte').textContent = '« ' + data.question_text + ' »';
    document.getElementById('clEliminationInfo').textContent    = '';

    clResetCheckboxes();
    clShowZone('clZoneReponse');
    clSetBanner('🤖 Akinator a répondu — à son tour !', 'akinator');
  } catch(e) {
    alert('Erreur : ' + e.message);
  }
  clLoading(false);
}


// ══════════════════════════════════════════════════════════════
//  Tour d'Akinator
// ══════════════════════════════════════════════════════════════
async function clTourAkinator() {
  clLoading(true);
  try {
    const r    = await fetch('/classique/akinator_turn/' + clSessionId);
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();

    if (data.type === 'question') {
      document.getElementById('clAkinatorQuestionTexte').textContent = data.question_text;
      clShowZone('clZoneAkinatorQuestion');
      clSetBanner('🤖 Akinator vous pose une question !', 'akinator');
    } else {
      document.getElementById('clAkinatorProposalPerso').textContent = data.personnage;
      clShowZone('clZoneAkinatorProposition');
      clSetBanner('🤔 Akinator fait une proposition !', 'akinator');
    }
  } catch(e) {
    alert('Erreur tour Akinator : ' + e.message);
  }
  clLoading(false);
}

async function clRepondreAkinator(reponse) {
  clLoading(true);
  try {
    const r = await fetch('/classique/player_answers', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: clSessionId, reponse})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();

    if (data.proposition) {
      document.getElementById('clAkinatorProposalPerso').textContent = data.proposition.personnage;
      clShowZone('clZoneAkinatorProposition');
      clSetBanner('🤔 Akinator fait une proposition !', 'akinator');
    } else {
      if (data.n_remaining !== undefined) clUpdateRemainingCount(data.n_remaining);
      clShowZone('clZoneJoueur');
      clSetBanner('🙋 Votre tour — posez une question !', 'joueur');
    }
  } catch(e) {
    alert('Erreur : ' + e.message);
  }
  clLoading(false);
}

async function clConfirmerPropositionAkinator(correct) {
  clLoading(true);
  try {
    const r = await fetch('/classique/confirm_akinator_proposal', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: clSessionId, correct})
    });
    if (!r.ok) throw new Error((await r.json()).detail);
    const data = await r.json();

    if (correct) {
      clAfficherFin('😮 Akinator a trouvé ! Il gagne !',
                    'Akinator a correctement deviné votre personnage.',
                    data.personnage_akinator);
    } else {
      clAfficherFin('🎉 Vous avez gagné ! Akinator a perdu !',
                    "Akinator s'est trompé — il n'a pas trouvé votre personnage.",
                    data.personnage_akinator);
    }
  } catch(e) {
    alert('Erreur : ' + e.message);
  }
  clLoading(false);
}


// ══════════════════════════════════════════════════════════════
//  Fin de partie classique
// ══════════════════════════════════════════════════════════════
function clAfficherFin(titre, message, nomAkinator) {
  document.getElementById('clEndTitle').textContent   = titre;
  document.getElementById('clEndMessage').textContent = message;
  document.getElementById('clNomAkinator').textContent = nomAkinator || '?';
  clShowScreen('clEndScreen');
}

</script>
</body>
</html>
"""

# ── Routes Qui est-ce ────────────────────────────────────────────────────────
@app.get("/animals")
def get_animals():
    """Retourne la liste des animaux et des questions depuis le CSV."""
    animaux, _, questions, _ = charger_donnees()
    return {"animaux": animaux, "questions": questions}


@app.post("/start_qui_est_ce")
def start_qui_est_ce(request: StartQECRequest):
    """Démarre une nouvelle partie de Qui est-ce."""
    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    total = sum(compteur_apparitions)
    if total == 0:
        proba_animaux = np.ones(len(animaux)) / len(animaux)
    else:
        proba_animaux = np.asarray([v / total for v in compteur_apparitions], dtype=float)

    # Akinator choisit son animal secret au hasard
    animal_akinator_idx = int(np.random.randint(0, len(animaux)))

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "animaux":                   animaux,
        "compteur_apparitions":      compteur_apparitions,
        "questions":                 questions,
        "donnees":                   donnees,
        "animal_joueur":             request.animal_joueur_index,
        "animal_akinator":           animal_akinator_idx,
        "proba_animaux":             proba_animaux,
        "reponses_donnees_akinator": {},
        "questions_pas_encore_posees": np.ones(len(questions), dtype=bool),
        "tour":                      "joueur",
        "question_akinator_courante": None,
        "animal_propose_akinator":   None,
    }

    return {
        "session_id": session_id,
        "animaux":    animaux,
        "questions":  questions,
        "tour":       "joueur",
    }


@app.post("/player_asks")
def player_asks(request: PlayerAsksRequest):
    """
    Le joueur pose une question à Akinator.
    Akinator répond en convertissant la valeur CSV de son animal secret
    en texte (ex. 0,69 → 'Plutôt oui' car plus proche de 0,75 que de 0,5).
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    if session["tour"] != "joueur":
        raise HTTPException(status_code=400, detail="Ce n'est pas le tour du joueur")

    q_idx          = request.question_index
    animal_aki_idx = session["animal_akinator"]
    valeur         = float(session["donnees"][q_idx, animal_aki_idx])
    reponse_idx    = valeur_vers_reponse_index(valeur)
    reponse_text   = REPONSES_POSSIBLES[reponse_idx]
    question_text  = session["questions"][q_idx]

    # C'est maintenant le tour d'Akinator
    session["tour"] = "akinator"

    return {
        "question_posee":  question_text,
        "akinator_reponse": reponse_text,
        "valeur":           valeur,
    }


@app.get("/akinator_turn/{session_id}")
def akinator_turn(session_id: str):
    """
    Calcule le prochain move d'Akinator :
    - S'il est sûr à > 0,90 → il propose son animal
    - Sinon → il pose la meilleure question (minimum d'information)
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[session_id]

    if session["tour"] != "akinator":
        raise HTTPException(status_code=400, detail="Ce n'est pas le tour d'Akinator")

    # Vérifier si Akinator est assez confiant pour proposer
    i_propose, proba = recherche_bonne_reponse_qec(session["proba_animaux"])

    if i_propose is not None:
        session["animal_propose_akinator"] = i_propose
        return {
            "type":        "proposal",
            "animal":      session["animaux"][i_propose],
            "animal_index": int(i_propose),
            "probabilite": float(proba),
        }

    # Sinon, choisir la meilleure question
    i_question = choix_meilleure_question(
        session["donnees"],
        session["proba_animaux"],
        session["questions_pas_encore_posees"]
    )

    if i_question is None:
        # Plus de questions disponibles → proposition forcée
        i_max = int(np.argmax(session["proba_animaux"]))
        session["animal_propose_akinator"] = i_max
        return {
            "type":        "proposal",
            "animal":      session["animaux"][i_max],
            "animal_index": i_max,
            "probabilite": float(session["proba_animaux"][i_max]),
        }

    session["question_akinator_courante"] = int(i_question)
    return {
        "type":          "question",
        "question_text": session["questions"][i_question],
        "question_index": int(i_question),
    }


@app.post("/player_answers_akinator")
def player_answers_akinator(request: PlayerAnswersAkinatorRequest):
    """
    Le joueur répond à la question d'Akinator sur son propre animal.
    Met à jour les probabilités d'Akinator (apprentissage bayésien).
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    if session["tour"] != "akinator":
        raise HTTPException(status_code=400, detail="Ce n'est pas le tour d'Akinator")
    if session["question_akinator_courante"] is None:
        raise HTTPException(status_code=400, detail="Pas de question en cours")
    if not (0 <= request.reponse < len(VALEURS_REPONSES)):
        raise HTTPException(status_code=400, detail="Réponse invalide")

    i_q = session["question_akinator_courante"]

    # Sauvegarder pour l'apprentissage en fin de partie
    session["reponses_donnees_akinator"][i_q] = request.reponse
    session["questions_pas_encore_posees"][i_q] = False

    # Mise à jour bayésienne des probabilités d'Akinator
    nouvelles_probas, _ = donner_proba_animaux_sachant_r(
        VALEURS_REPONSES[request.reponse],
        session["donnees"],
        session["proba_animaux"]
    )
    session["proba_animaux"] = nouvelles_probas[i_q]
    session["proba_animaux"] += 1e-6
    session["proba_animaux"] /= np.sum(session["proba_animaux"])

    # Retour au tour du joueur
    session["tour"] = "joueur"
    session["question_akinator_courante"] = None

    # Vérifier si Akinator est maintenant assez confiant pour proposer
    i_propose, proba = recherche_bonne_reponse_qec(session["proba_animaux"])
    if i_propose is not None:
        session["animal_propose_akinator"] = i_propose
        session["tour"] = "akinator"  # reste en mode akinator pour la proposition
        return {
            "tour":       "akinator",
            "proba_max":  float(proba),
            "proposition": {
                "animal":      session["animaux"][i_propose],
                "animal_index": int(i_propose),
                "probabilite": float(proba),
            }
        }

    return {
        "tour":      "joueur",
        "proba_max": float(np.max(session["proba_animaux"])),
        "proposition": None,
    }


@app.post("/confirm_akinator_proposal")
def confirm_akinator_proposal(request: ConfirmAkinatorProposalRequest):
    """
    Le joueur confirme si la proposition d'Akinator est correcte.
    - correct=True  → animal_joueur = animal_proposé → apprentissage → fin (Akinator gagne)
    - correct=False → on attend que le joueur déclare son animal via /declare_joueur_animal
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    if session["animal_propose_akinator"] is None:
        raise HTTPException(status_code=400, detail="Pas de proposition en cours")

    animal_propose  = session["animal_propose_akinator"]
    animal_akinator = session["animal_akinator"]

    if request.correct:
        # Le joueur confirme → on sait que son animal = animal proposé
        session["animal_joueur"] = animal_propose
        nom_joueur   = session["animaux"][animal_propose]
        nom_akinator = session["animaux"][animal_akinator]
        terminer_partie_et_apprendre(session)
        del sessions[request.session_id]
        return {
            "correct":         True,
            "animal_joueur":   nom_joueur,
            "animal_akinator": nom_akinator,
        }
    else:
        # Akinator s'est trompé → on garde la session ouverte pour la déclaration
        session["animal_propose_akinator"] = None
        session["tour"] = "joueur"
        return {
            "correct": False,
        }


@app.post("/declare_joueur_animal")
def declare_joueur_animal(request: DeclareJoueurAnimalRequest):
    """
    Le joueur déclare son animal pour l'apprentissage.
    Utilisé après : erreur d'Akinator, victoire du joueur, victoire d'Akinator solo.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    session["animal_joueur"] = request.animal_index
    animal_akinator = session["animal_akinator"]
    nom_joueur   = session["animaux"][request.animal_index]
    nom_akinator = session["animaux"][animal_akinator]

    terminer_partie_et_apprendre(session)
    del sessions[request.session_id]

    return {
        "titre":           "📚 Merci ! Akinator a appris.",
        "message":         f"Votre animal était {nom_joueur}. Akinator va s'en souvenir !",
        "animal_joueur":   nom_joueur,
        "animal_akinator": nom_akinator,
    }


@app.post("/player_proposes")
def player_proposes(request: PlayerProposesRequest):
    """
    Le joueur propose l'animal d'Akinator.
    - Correct  → session maintenue pour que le joueur déclare son animal (apprentissage)
    - Incorrect → session maintenue pour Akinator solo (/akinator_solo_propose)
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    animal_propose  = request.animal_index
    animal_akinator = session["animal_akinator"]
    correct         = (animal_propose == animal_akinator)
    nom_akinator    = session["animaux"][animal_akinator]

    return {
        "correct":             correct,
        "animal_akinator":     nom_akinator,
        "animal_akinator_nom": nom_akinator,
    }


@app.get("/akinator_solo_propose/{session_id}")
def akinator_solo_propose(session_id: str):
    """
    Akinator propose son meilleur candidat actuel après erreur du joueur.
    Utilise la distribution de probabilité déjà construite (sans nouvelles réponses).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[session_id]

    i_max = int(np.argmax(session["proba_animaux"]))
    proba = float(session["proba_animaux"][i_max])
    session["animal_propose_akinator"] = i_max

    return {
        "animal":       session["animaux"][i_max],
        "animal_index": i_max,
        "probabilite":  proba,
    }


class AkinatorSoloCorrectRequest(BaseModel):
    session_id: str

@app.post("/akinator_solo_correct")
def akinator_solo_correct(request: AkinatorSoloCorrectRequest):
    """
    Le joueur confirme qu'Akinator a trouvé en solo.
    L'animal du joueur = animal_propose_akinator (déjà connu).
    → Apprentissage immédiat + fermeture de session.
    """
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    animal_propose = session.get("animal_propose_akinator")
    if animal_propose is None:
        raise HTTPException(status_code=400, detail="Pas de proposition en cours")

    # On sait que l'animal du joueur = la proposition d'Akinator
    session["animal_joueur"] = animal_propose
    nom_joueur   = session["animaux"][animal_propose]
    nom_akinator = session["animaux"][session["animal_akinator"]]

    terminer_partie_et_apprendre(session)
    del sessions[request.session_id]

    return {
        "animal_joueur":   nom_joueur,
        "animal_akinator": nom_akinator,
    }


class PartieNulleRequest(BaseModel):
    session_id: str

@app.post("/partie_nulle")
def partie_nulle(request: PartieNulleRequest):
    """Fin de partie en match nul — ni le joueur ni Akinator n'ont trouvé."""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]
    nom_akinator = session["animaux"][session["animal_akinator"]]
    del sessions[request.session_id]
    return {"animal_akinator": nom_akinator}


class AddQuestionRequest(BaseModel):
    question: str
    valeurs: list  # liste de float, une par animal

class UpdateQuestionRequest(BaseModel):
    index: int
    question: str
    valeurs: list

class DeleteQuestionRequest(BaseModel):
    index: int


# ── Routes Version Classique ─────────────────────────────────────────────────

class StartClassiqueRequest(BaseModel):
    personnage_joueur_index: int

class PlayerAsksClassiqueRequest(BaseModel):
    session_id: str
    attr_indices: list

class PlayerAnswersClassiqueRequest(BaseModel):
    session_id: str
    reponse: int

class PlayerProposesClassiqueRequest(BaseModel):
    session_id: str
    personnage_index: int

class ConfirmClassiqueRequest(BaseModel):
    session_id: str
    correct: bool

@app.get("/classique/personnages")
def get_personnages_classique():
    """Retourne la liste des personnages et attributs du CSV classique."""
    personnages, attributs, _ = charger_donnees_classique()
    return {"personnages": personnages, "attributs": attributs}

@app.post("/classique/start")
def start_classique(request: StartClassiqueRequest):
    """Démarre une nouvelle partie classique."""
    personnages, attributs, donnees = charger_donnees_classique()
    n = len(personnages)
    if not (0 <= request.personnage_joueur_index < n):
        raise HTTPException(status_code=400, detail="Index de personnage invalide")
    personnage_akinator = _random.randint(0, n - 1)
    sid = str(uuid.uuid4())
    sessions_classique[sid] = {
        "personnages":            personnages,
        "attributs":              attributs,
        "donnees":                donnees,
        "personnage_joueur":      request.personnage_joueur_index,
        "personnage_akinator":    personnage_akinator,
        "remaining_akinator":     list(range(n)),
        "attrs_poses_akinator":   set(),
        "attrs_poses_joueur":     set(),
        "question_akinator_courante": None,
        "personnage_propose_akinator": None,
    }
    return {"session_id": sid, "personnages": personnages, "attributs": attributs}

@app.post("/classique/player_asks")
def player_asks_classique(request: PlayerAsksClassiqueRequest):
    """Le joueur pose une question (OR logique sur les attrs) — Akinator répond Oui/Non."""
    if request.session_id not in sessions_classique:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions_classique[request.session_id]
    if not request.attr_indices:
        raise HTTPException(status_code=400, detail="Aucun attribut sélectionné")
    attrs   = request.attr_indices
    donnees = session["donnees"]
    n       = len(session["personnages"])
    aki_idx = session["personnage_akinator"]
    reponse_oui = any(donnees[a][aki_idx] == 1 for a in attrs)
    for a in attrs:
        session["attrs_poses_joueur"].add(a)
    if reponse_oui:
        compatible = [c for c in range(n) if any(donnees[a][c] == 1 for a in attrs)]
    else:
        compatible = [c for c in range(n) if all(donnees[a][c] == 0 for a in attrs)]
    return {
        "question_text":      formater_question_classique(session["attributs"], attrs),
        "reponse":            "Oui" if reponse_oui else "Non",
        "compatible_indices": compatible,
    }

@app.get("/classique/akinator_turn/{session_id}")
def akinator_turn_classique(session_id: str):
    """Akinator choisit la meilleure question ou propose un personnage."""
    if session_id not in sessions_classique:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions_classique[session_id]
    remaining   = session["remaining_akinator"]
    n_remaining = len(remaining)
    if n_remaining <= 2:
        idx = remaining[0]
        session["personnage_propose_akinator"] = idx
        return {"type": "proposal", "personnage": session["personnages"][idx], "personnage_index": idx}
    attrs_dispo = set(range(len(session["attributs"]))) - session["attrs_poses_akinator"]
    if not attrs_dispo:
        idx = remaining[0]
        session["personnage_propose_akinator"] = idx
        return {"type": "proposal", "personnage": session["personnages"][idx], "personnage_index": idx}
    max_size = 3 if n_remaining > 6 else (2 if n_remaining > 3 else 1)
    combo = meilleure_combinaison_classique(session["donnees"], remaining, attrs_dispo, max_size)
    session["question_akinator_courante"] = combo
    return {
        "type":          "question",
        "question_text": formater_question_classique(session["attributs"], combo),
        "attr_indices":  combo,
    }

@app.post("/classique/player_answers")
def player_answers_classique(request: PlayerAnswersClassiqueRequest):
    """Le joueur répond Oui(1)/Non(0) — mise à jour des candidats restants d'Akinator."""
    if request.session_id not in sessions_classique:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions_classique[request.session_id]
    if session["question_akinator_courante"] is None:
        raise HTTPException(status_code=400, detail="Pas de question en cours")
    combo       = session["question_akinator_courante"]
    reponse_oui = bool(request.reponse)
    donnees     = session["donnees"]
    for a in combo:
        session["attrs_poses_akinator"].add(a)
    if reponse_oui:
        session["remaining_akinator"] = [
            c for c in session["remaining_akinator"]
            if any(donnees[a][c] == 1 for a in combo)
        ]
    else:
        session["remaining_akinator"] = [
            c for c in session["remaining_akinator"]
            if all(donnees[a][c] == 0 for a in combo)
        ]
    session["question_akinator_courante"] = None
    n_rem = len(session["remaining_akinator"])
    if n_rem <= 2 and n_rem > 0:
        idx = session["remaining_akinator"][0]
        session["personnage_propose_akinator"] = idx
        return {
            "tour": "akinator", "n_remaining": n_rem,
            "proposition": {"personnage": session["personnages"][idx], "personnage_index": idx}
        }
    return {"tour": "joueur", "n_remaining": n_rem, "proposition": None}

@app.post("/classique/player_proposes")
def player_proposes_classique(request: PlayerProposesClassiqueRequest):
    """Le joueur propose le personnage d'Akinator."""
    if request.session_id not in sessions_classique:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions_classique[request.session_id]
    aki = session["personnage_akinator"]
    correct = (request.personnage_index == aki)
    return {
        "correct":                   correct,
        "personnage_akinator":       session["personnages"][aki],
        "personnage_akinator_index": aki,
        "personnage_joueur":         session["personnages"][session["personnage_joueur"]],
    }

@app.post("/classique/confirm_akinator_proposal")
def confirm_akinator_proposal_classique(request: ConfirmClassiqueRequest):
    """Le joueur confirme ou infirme la proposition d'Akinator."""
    if request.session_id not in sessions_classique:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions_classique[request.session_id]
    nom_joueur = session["personnages"][session["personnage_joueur"]]
    nom_aki    = session["personnages"][session["personnage_akinator"]]
    if request.correct:
        del sessions_classique[request.session_id]
        return {"correct": True, "personnage_akinator": nom_aki}
    else:
        del sessions_classique[request.session_id]
        return {"correct": False, "personnage_akinator": nom_aki}


# ── Routes Admin Classique ───────────────────────────────────────────────────

class UpdatePersonnageClassiqueRequest(BaseModel):
    index:   int
    nom:     str
    valeurs: list   # liste de int (0 ou 1), une valeur par attribut

@app.get("/classique/admin/data")
def classique_admin_get_data():
    """Retourne personnages, attributs ET matrice pour le panneau admin classique."""
    personnages, attributs, donnees = charger_donnees_classique()
    # donnees[attr_idx][perso_idx] → on transpose pour l'affichage
    return {
        "personnages": personnages,
        "attributs":   attributs,
        "donnees":     donnees,   # liste de listes : donnees[attr][perso]
    }

@app.put("/classique/admin/update_personnage")
def classique_admin_update_personnage(request: UpdatePersonnageClassiqueRequest):
    """Met à jour le nom et les valeurs d'un personnage dans le CSV classique."""
    personnages, attributs, donnees = charger_donnees_classique()
    n = len(personnages)
    if not (0 <= request.index < n):
        raise HTTPException(status_code=400, detail="Index de personnage invalide")
    if len(request.valeurs) != len(attributs):
        raise HTTPException(
            status_code=400,
            detail=f"Il faut {len(attributs)} valeurs, reçu {len(request.valeurs)}"
        )
    for v in request.valeurs:
        if v not in (0, 1):
            raise HTTPException(status_code=400, detail=f"Valeur invalide : {v} (attendu 0 ou 1)")

    # Appliquer les modifications
    personnages[request.index] = request.nom.strip()
    for attr_idx, val in enumerate(request.valeurs):
        donnees[attr_idx][request.index] = val

    # Réécrire le CSV
    import csv as _csv
    with open(CSV_CLASSIQUE_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = _csv.writer(f)
        writer.writerow(['Attribut'] + personnages)
        for attr_idx, attr in enumerate(attributs):
            writer.writerow([attr] + donnees[attr_idx])

    return {"ok": True, "nom": personnages[request.index]}


# ── Routes Admin ─────────────────────────────────────────────────────────────
@app.get("/admin/data")
def admin_get_data():
    """Retourne animaux, questions ET données (valeurs) pour le panneau admin."""
    animaux, _, questions, donnees = charger_donnees()
    return {
        "animaux":   animaux,
        "questions": questions,
        "donnees":   donnees.tolist(),   # liste de listes de float
    }


@app.post("/admin/add_question")
def admin_add_question(request: AddQuestionRequest):
    """Ajoute une nouvelle question avec ses valeurs initiales."""
    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    if len(request.valeurs) != len(animaux):
        raise HTTPException(status_code=400,
            detail=f"Il faut {len(animaux)} valeurs, reçu {len(request.valeurs)}")
    for v in request.valeurs:
        if not (0.0 <= float(v) <= 1.0):
            raise HTTPException(status_code=400,
                detail=f"Valeur hors [0,1] : {v}")
    if request.question.strip() in questions:
        raise HTTPException(status_code=400, detail="Cette question existe déjà")

    nouvelle_ligne    = np.array(request.valeurs, dtype=float).reshape(1, -1)
    nouvelles_donnees = np.vstack([donnees, nouvelle_ligne])
    nouvelles_questions = questions + [request.question.strip()]

    sauvegarde_csv(animaux, compteur_apparitions, nouvelles_questions, nouvelles_donnees)
    return {
        "ok": True,
        "question_index": len(nouvelles_questions) - 1,
        "total_questions": len(nouvelles_questions),
    }


@app.put("/admin/update_question")
def admin_update_question(request: UpdateQuestionRequest):
    """Modifie le texte et/ou les valeurs d'une question existante."""
    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    if not (0 <= request.index < len(questions)):
        raise HTTPException(status_code=400, detail="Index hors limites")
    if len(request.valeurs) != len(animaux):
        raise HTTPException(status_code=400,
            detail=f"Il faut {len(animaux)} valeurs, reçu {len(request.valeurs)}")
    for v in request.valeurs:
        if not (0.0 <= float(v) <= 1.0):
            raise HTTPException(status_code=400, detail=f"Valeur hors [0,1] : {v}")

    questions[request.index]        = request.question.strip()
    donnees[request.index]          = np.array(request.valeurs, dtype=float)

    sauvegarde_csv(animaux, compteur_apparitions, questions, donnees)
    return {"ok": True}


@app.delete("/admin/delete_question")
def admin_delete_question(request: DeleteQuestionRequest):
    """Supprime une question (et sa ligne de valeurs) du CSV."""
    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    if not (0 <= request.index < len(questions)):
        raise HTTPException(status_code=400, detail="Index hors limites")

    questions.pop(request.index)
    nouvelles_donnees = np.delete(donnees, request.index, axis=0)

    sauvegarde_csv(animaux, compteur_apparitions, questions, nouvelles_donnees)
    return {"ok": True, "total_questions": len(questions)}


# ── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    # Pour lancer : python "D:\Documents\A_conserver\Cours\Programmation\Labo IA\Akinator\Akinator version Qui est ce\main.py"