"""
API FastAPI pour Akinator avec Google Drive
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

# Constantes
REPONSES_POSSIBLES = ["Oui", "Plutôt oui", "Je ne sais pas (ou entre oui et non)", "Plutôt non", "Non"]
VALEURS_REPONSES = [1, 0.75, 0.5, 0.25, 0]
SEUIL = 0.7 # Seuil à partir duquel on estime que la réponse est bonne.

# Configuration Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
GDRIVE_FILE_NAME = "akinator_animaux.csv"
GDRIVE_SUGGESTIONS_FILE = "contenu_a_ajouter.csv"

# Stockage des sessions en mémoire
sessions: Dict[str, dict] = {}

app = FastAPI(title="Akinator API")

try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    pass  # Le dossier static n'existe peut-être pas

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic
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

# Fonctions Google Drive
def get_drive_service():
    """Initialise le service Google Drive"""
    try:
        creds_json = os.getenv('GOOGLE_CREDENTIALS')
        if not creds_json:
            raise ValueError("GOOGLE_CREDENTIALS non définie")

        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )

        service = build('drive', 'v3', credentials=credentials)
        return service
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Google Drive: {str(e)}")

def find_file_id(service, filename):
    """Trouve l'ID du fichier sur Google Drive"""
    try:
        results = service.files().list(
            q=f"name='{filename}' and trashed=false",
            spaces='drive',
            fields='files(id, name)'
        ).execute()

        files = results.get('files', [])
        if not files:
            return None
        return files[0]['id']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur recherche fichier: {str(e)}")

def download_csv_from_drive():
    """Télécharge le CSV depuis Google Drive"""
    try:
        service = get_drive_service()
        file_id = find_file_id(service, GDRIVE_FILE_NAME)

        if not file_id:
            raise HTTPException(status_code=404, detail=f"Fichier '{GDRIVE_FILE_NAME}' non trouvé sur Google Drive")

        request = service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()
        downloader = MediaIoBaseDownload(file_handle, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()

        file_handle.seek(0)
        content = file_handle.read().decode('utf-8')
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur téléchargement: {str(e)}")

def upload_csv_to_drive(content, filename=GDRIVE_FILE_NAME):
    """Upload le CSV vers Google Drive"""
    try:
        service = get_drive_service()
        file_id = find_file_id(service, filename)

        file_metadata = {'name': filename}
        media = MediaIoBaseUpload(
            io.BytesIO(content.encode('utf-8')),
            mimetype='text/csv',
            resumable=True
        )

        if file_id:
            service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
        else:
            service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur upload: {str(e)}")

def charger_donnees():
    """Charge les données depuis Google Drive"""
    try:
        csv_content = download_csv_from_drive()

        reader = csv.reader(io.StringIO(csv_content))
        animaux = next(reader)[1:]
        compteur_apparitions = [int(val) for val in next(reader)[1:]]
        questions = []
        donnees = []

        for ligne in reader:
            questions.append(ligne[0])
            donnees.append([float(val) for val in ligne[1:]])

        donnees = np.asarray(donnees)
        return animaux, compteur_apparitions, questions, donnees
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur chargement données: {str(e)}")

def sauvegarde_csv(animaux, compteur_apparitions, questions, donnees):
    """Sauvegarde le CSV sur Google Drive"""
    try:
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow(["Question"] + animaux)
        writer.writerow(["Compteur d'apparitions"] + compteur_apparitions)

        for i in range(len(questions)):
            writer.writerow([questions[i]] + [f"{valeur:.3f}" for valeur in donnees[i]])

        csv_content = output.getvalue()
        upload_csv_to_drive(csv_content, GDRIVE_FILE_NAME)
    except Exception as e:
        print(f"Erreur sauvegarde: {str(e)}")

def sauvegarder_suggestion(animal, question, reponses_donnees, questions):
    """Sauvegarde une suggestion d'animal dans contenu_a_ajouter.csv"""
    try:
        service = get_drive_service()
        file_id = find_file_id(service, GDRIVE_SUGGESTIONS_FILE)

        # Lire toutes les lignes existantes
        lignes_existantes = []
        if file_id:
            try:
                request = service.files().get_media(fileId=file_id)
                file_handle = io.BytesIO()
                downloader = MediaIoBaseDownload(file_handle, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                file_handle.seek(0)
                existing_content = file_handle.read().decode('utf-8')

                # Parser le CSV existant
                reader = csv.reader(io.StringIO(existing_content))
                lignes_existantes = list(reader)

            except Exception as e:
                pass

        # Créer le nouveau contenu
        output = io.StringIO()
        writer = csv.writer(output)

        # Si pas de contenu existant, créer l'en-tête
        if not lignes_existantes:
            writer.writerow(["Animal", "Question proposée"])
        else:
            # Écrire toutes les lignes existantes
            for idx, ligne in enumerate(lignes_existantes):
                writer.writerow(ligne)

        # Ajouter la nouvelle suggestion
        writer.writerow([animal, question])

        # Récupérer le contenu final
        csv_content = output.getvalue()

        # Upload
        upload_csv_to_drive(csv_content, GDRIVE_SUGGESTIONS_FILE)
        return True
    except Exception as e:
        return False

# Fonctions de calcul
def donner_proba_animaux_sachant_r(r, donnees, proba_animaux):
    numerateurs = np.maximum(1 - abs(r - donnees), 0.05) * proba_animaux
    denominateur = np.sum(numerateurs, axis=1)
    denominateur[denominateur == 0] = 1
    return np.divide(numerateurs, denominateur.reshape(-1, 1)), denominateur

def calcul_IM(donnees, proba_animaux):
    IM = 0
    for r in VALEURS_REPONSES:
        proba_animaux_sachant_r, p_r = donner_proba_animaux_sachant_r(r, donnees, proba_animaux)
        h_r = np.sum(np.where(proba_animaux_sachant_r > 0,
                              -proba_animaux_sachant_r * np.log2(proba_animaux_sachant_r), 0), axis=1)
        IM += h_r * p_r
    return IM

def choix_meilleure_question(donnees, proba_animaux, questions_pas_encore_posees):
    IM = calcul_IM(donnees, proba_animaux)
    if any(questions_pas_encore_posees):
        return np.argmin(np.where(questions_pas_encore_posees, IM, np.inf))
    else:
        return None

def recherche_bonne_reponse(proba_animaux, animaux):
    i_premier, i_second, *_ = np.argsort(proba_animaux)[::-1]
    if proba_animaux[i_premier] > SEUIL:
        return i_premier, proba_animaux[i_premier]
    else:
        return None, None

def actualisation_valeurs_theoriques(donnees, indice_animal, reponses_donnees, alpha=0.1):
    for question, reponse in reponses_donnees.items():
        donnees[question, indice_animal] += alpha * (VALEURS_REPONSES[reponse] - donnees[question, indice_animal])

# Route pour l'interface HTML
@app.get("/", response_class=HTMLResponse)
def get_interface():
    return """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Akinator - Devine ton animal</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
                text-align: center;
            }

            h1 {
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }

            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }

            .question-container {
                background: #f8f9ff;
                border-radius: 15px;
                padding: 30px;
                margin: 30px 0;
                display: none;
            }

            .question-container.active {
                display: block;
            }

            .question-number {
                color: #764ba2;
                font-weight: bold;
                margin-bottom: 15px;
                font-size: 1.1em;
            }

            .question-text {
                font-size: 1.4em;
                color: #333;
                margin-bottom: 30px;
                font-weight: 500;
            }

            .buttons-container {
                display: flex;
                flex-direction: column;
                gap: 12px;
            }

            button {
                padding: 15px 30px;
                font-size: 1.1em;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                transition: all 0.3s ease;
                font-weight: 600;
                color: white;
            }

            .btn-start {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-size: 1.3em;
                padding: 20px 40px;
            }

            .btn-start:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }

            .btn-answer {
                background: #667eea;
            }

            .btn-answer:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }

            .btn-answer:active {
                transform: translateY(0);
            }

            .guess-container {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border-radius: 15px;
                padding: 30px;
                margin: 30px 0;
                display: none;
            }

            .guess-container.active {
                display: block;
            }

            .guess-title {
                font-size: 1.5em;
                margin-bottom: 20px;
            }

            .guess-animal {
                font-size: 2em;
                font-weight: bold;
                margin: 20px 0;
            }

            .guess-probability {
                font-size: 1.2em;
                opacity: 0.9;
                margin-bottom: 30px;
            }

            .btn-confirm {
                background: white;
                color: #f5576c;
                margin: 5px;
            }

            .btn-confirm:hover {
                background: #f0f0f0;
            }

            .welcome-screen {
                display: block;
            }

            .welcome-screen.hidden {
                display: none;
            }

            .loading {
                display: none;
                margin: 20px 0;
            }

            .loading.active {
                display: block;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .success-message {
                background: #4caf50;
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                display: none;
            }

            .success-message.active {
                display: block;
            }

            input[type="text"] {
                width: 100%;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 10px;
                border: none;
                font-size: 1.1em;
                color: #333;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔮 Akinator</h1>
            <p class="subtitle">Pense à un animal, je vais le deviner !</p>

            <div class="welcome-screen" id="welcomeScreen">
                <p style="margin: 30px 0; color: #666; font-size: 1.1em;">
                    Je vais te poser quelques questions pour deviner l'animal auquel tu penses.
                </p>
                <button class="btn-start" onclick="startGame()">Commencer</button>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">Chargement...</p>
            </div>

            <div class="question-container" id="questionContainer">
                <div class="question-number" id="questionNumber"></div>
                <div class="question-text" id="questionText"></div>
                <div class="buttons-container" id="buttonsContainer"></div>
            </div>

            <div class="guess-container" id="guessContainer">
                <div class="guess-title">Je pense à...</div>
                <div class="guess-animal" id="guessAnimal"></div>
                <div class="guess-probability" id="guessProbability"></div>
                <div>
                    <button class="btn-confirm" onclick="confirmGuess(true)">✓ Oui, c'est ça !</button>
                    <button class="btn-confirm" onclick="confirmGuess(false)">✗ Non, continue</button>
                </div>
            </div>

            <div class="success-message" id="successMessage">
                🎉 Super ! J'ai trouvé ! Merci d'avoir joué !
            </div>

            <div class="guess-container" id="suggestionContainer" style="background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);">
                <div class="guess-title">Je ne trouve pas... Aide-moi ! 😅</div>
                <p style="margin-bottom: 20px; font-size: 1.1em;">À quel animal pensais-tu ?</p>
                <input type="text" id="animalInput" placeholder="Ex: Dauphin">
                <p style="margin-bottom: 10px; font-size: 1.1em;">Propose une question pour distinguer cet animal :</p>
                <input type="text" id="questionInput" placeholder="Ex: Vit-il dans l'eau ?">
                <button class="btn-confirm" onclick="submitSuggestion()">📝 Envoyer ma suggestion</button>
            </div>
        </div>

        <script>
            let sessionId = null;

            async function startGame() {
                showLoading();
                hideWelcome();

                try {
                    const response = await fetch('/start', {
                        method: 'POST'
                    });
                    const data = await response.json();

                    console.log('Données reçues:', data); // Pour debug

                    if (!data.reponses_possibles) {
                        console.error('reponses_possibles est manquant!');
                        alert('Erreur: données incomplètes du serveur');
                        return;
                    }

                    sessionId = data.session_id;
                    displayQuestion(data.question, data.question_number, data.reponses_possibles);
                } catch (error) {
                    console.error('Erreur complète:', error);
                    alert('Erreur lors du démarrage du jeu : ' + error);
                }
            }

            function displayQuestion(question, number, reponses) {
                hideLoading();
                hideGuess();

                document.getElementById('questionNumber').textContent = `Question n°${number}`;
                document.getElementById('questionText').textContent = question;

                const buttonsContainer = document.getElementById('buttonsContainer');
                buttonsContainer.innerHTML = '';

                reponses.forEach((reponse, index) => {
                    const button = document.createElement('button');
                    button.className = 'btn-answer';
                    button.textContent = reponse;
                    button.onclick = () => answerQuestion(index);
                    buttonsContainer.appendChild(button);
                });

                showQuestion();
            }

            async function answerQuestion(reponseIndex) {
                showLoading();
                hideQuestion();

                try {
                    const response = await fetch('/answer', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            session_id: sessionId,
                            reponse: reponseIndex
                        })
                    });
                    const data = await response.json();

                    if (data.is_final) {
                        displayGuess(data.animal, data.probabilite);
                    } else {
                        displayQuestion(data.question, data.question_number, ['Oui', 'Plutôt oui', 'Je ne sais pas', 'Plutôt non', 'Non']);
                    }
                } catch (error) {
                    alert('Erreur : ' + error);
                }
            }

            function displayGuess(animal, probabilite) {
                hideLoading();
                hideQuestion();

                document.getElementById('guessAnimal').textContent = animal;
                document.getElementById('guessProbability').textContent = `Probabilité : ${(probabilite * 100).toFixed(1)}%`;

                showGuess();
            }

            async function confirmGuess(correct) {
                showLoading();
                hideGuess();

                try {
                    const response = await fetch('/confirm', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            session_id: sessionId,
                            correct: correct
                        })
                    });
                    const data = await response.json();

                    hideLoading();

                    if (data.need_suggestion) {
                        showSuggestion();
                    } else if (data.message && !data.is_final) {
                        showSuccess();
                        setTimeout(() => {
                            location.reload();
                        }, 3000);
                    } else if (data.is_final) {
                        displayGuess(data.animal, data.probabilite);
                    } else {
                        displayQuestion(data.question, data.question_number, ['Oui', 'Plutôt oui', 'Je ne sais pas', 'Plutôt non', 'Non']);
                    }
                } catch (error) {
                    alert('Erreur : ' + error);
                }
            }

            async function submitSuggestion() {
                const animal = document.getElementById('animalInput').value.trim();
                const question = document.getElementById('questionInput').value.trim();

                if (!animal || !question) {
                    alert('Merci de remplir les deux champs !');
                    return;
                }

                showLoading();
                hideSuggestion();

                try {
                    const response = await fetch('/suggest', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            session_id: sessionId,
                            animal: animal,
                            question: question
                        })
                    });
                    const data = await response.json();

                    hideLoading();
                    document.getElementById('successMessage').textContent = data.message;
                    showSuccess();
                    setTimeout(() => {
                        location.reload();
                    }, 3000);
                } catch (error) {
                    alert('Erreur : ' + error);
                }
            }

            function showWelcome() {
                document.getElementById('welcomeScreen').classList.remove('hidden');
            }

            function hideWelcome() {
                document.getElementById('welcomeScreen').classList.add('hidden');
            }

            function showLoading() {
                document.getElementById('loading').classList.add('active');
            }

            function hideLoading() {
                document.getElementById('loading').classList.remove('active');
            }

            function showQuestion() {
                document.getElementById('questionContainer').classList.add('active');
            }

            function hideQuestion() {
                document.getElementById('questionContainer').classList.remove('active');
            }

            function showGuess() {
                document.getElementById('guessContainer').classList.add('active');
            }

            function hideGuess() {
                document.getElementById('guessContainer').classList.remove('active');
            }

            function showSuccess() {
                document.getElementById('successMessage').classList.add('active');
            }

            function showSuggestion() {
                document.getElementById('suggestionContainer').classList.add('active');
            }

            function hideSuggestion() {
                document.getElementById('suggestionContainer').classList.remove('active');
            }
        </script>
    </body>
    </html>
    """

# Routes API
@app.post("/start", response_model=SessionResponse)
def start_session():
    """Démarre une nouvelle session Akinator"""
    session_id = str(uuid.uuid4())

    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    apparitions_totales = sum(compteur_apparitions)
    proba_animaux = np.asarray([val / apparitions_totales for val in compteur_apparitions])

    sessions[session_id] = {
        "animaux": animaux,
        "compteur_apparitions": compteur_apparitions,
        "questions": questions,
        "donnees": donnees,
        "proba_animaux": proba_animaux,
        "reponses_donnees": {},
        "questions_pas_encore_posees": np.ones(len(questions), dtype=bool),
        "compteur_question": 0,
        "echecs_consecutifs": 0
    }

    i_question = choix_meilleure_question(donnees, proba_animaux, sessions[session_id]["questions_pas_encore_posees"])

    if i_question is None:
        raise HTTPException(status_code=500, detail="Aucune question disponible")

    sessions[session_id]["question_courante"] = i_question
    sessions[session_id]["compteur_question"] = 1

    return SessionResponse(
        session_id=session_id,
        question=questions[i_question],
        question_number=1,
        reponses_possibles=list(REPONSES_POSSIBLES)
    )

@app.post("/answer")
def answer_question(request: AnswerRequest):
    """Répond à une question et obtient la suivante ou la réponse finale"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")

    session = sessions[request.session_id]
    i_question = session["question_courante"]

    if not (0 <= request.reponse < len(VALEURS_REPONSES)):
        raise HTTPException(status_code=400, detail="Réponse invalide")

    session["reponses_donnees"][i_question] = request.reponse
    session["questions_pas_encore_posees"][i_question] = False

    session["proba_animaux"] = donner_proba_animaux_sachant_r(
        VALEURS_REPONSES[request.reponse],
        session["donnees"],
        session["proba_animaux"]
    )[0][i_question]

    session["proba_animaux"] += 1e-6
    session["proba_animaux"] /= np.sum(session["proba_animaux"])

    indice_meilleur_animal, proba = recherche_bonne_reponse(session["proba_animaux"], session["animaux"])

    if indice_meilleur_animal is not None:
        session["animal_propose"] = indice_meilleur_animal
        return GuessResponse(
            animal=session["animaux"][indice_meilleur_animal],
            probabilite=float(proba),
            is_final=True
        )

    i_question = choix_meilleure_question(
        session["donnees"],
        session["proba_animaux"],
        session["questions_pas_encore_posees"]
    )

    if i_question is None:
        i_max = np.argmax(session["proba_animaux"])
        session["animal_propose"] = i_max
        return GuessResponse(
            animal=session["animaux"][i_max],
            probabilite=float(session["proba_animaux"][i_max]),
            is_final=True
        )

    session["question_courante"] = i_question
    session["compteur_question"] += 1

    return GuessResponse(
        animal="",
        probabilite=0.0,
        is_final=False,
        question=session["questions"][i_question],
        question_number=session["compteur_question"]
    )

@app.post("/confirm")
def confirm_guess(request: ConfirmRequest):
    """Confirme si la réponse était correcte"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")

    session = sessions[request.session_id]

    if "animal_propose" not in session:
        raise HTTPException(status_code=400, detail="Aucun animal n'a été proposé")

    if request.correct:
        indice_animal = session["animal_propose"]
        session["compteur_apparitions"][indice_animal] += 1
        actualisation_valeurs_theoriques(
            session["donnees"],
            indice_animal,
            session["reponses_donnees"]
        )

        sauvegarde_csv(
            session["animaux"],
            session["compteur_apparitions"],
            session["questions"],
            session["donnees"]
        )

        del sessions[request.session_id]
        return {"message": "Parfait ! Merci d'avoir joué !"}
    else:
        # COMPTEUR D'ÉCHECS - ICI !!!
        session["echecs_consecutifs"] += 1
        session["proba_animaux"][session["animal_propose"]] = 0
        del session["animal_propose"]

        # VÉRIFICATION DES 3 ÉCHECS - ICI !!!
        if session["echecs_consecutifs"] >= 3:
            return {
                "need_suggestion": True,
                "message": "Je ne trouve pas... Peux-tu m'aider ?"
            }


        i_question = choix_meilleure_question(
        session["donnees"],
        session["proba_animaux"],
        session["questions_pas_encore_posees"]
        )

    if i_question is None:
        i_max = np.argmax(session["proba_animaux"])
        session["animal_propose"] = i_max
        return GuessResponse(
            animal=session["animaux"][i_max],
            probabilite=float(session["proba_animaux"][i_max]),
            is_final=True
        )

    session["question_courante"] = i_question
    session["compteur_question"] += 1

    return GuessResponse(
        animal="",
        probabilite=0.0,
        is_final=False,
        question=session["questions"][i_question],
        question_number=session["compteur_question"]
    )

@app.post("/suggest")
def submit_suggestion(request: SuggestionRequest):
    """SAUVEGARDE LA SUGGESTION - ICI !!!"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session non trouvée")
    session = sessions[request.session_id]

    # APPEL À LA FONCTION DE SAUVEGARDE - ICI !!!
    success = sauvegarder_suggestion(
        request.animal,
        request.question,
        session["reponses_donnees"],
        session["questions"]
    )

    del sessions[request.session_id]

    if success:
        return {"message": f"Merci ! J'ai appris que '{request.animal}' existe. Ta suggestion a été enregistrée !"}
    else:
        return {"message": "Merci pour ta suggestion !"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    # Pour lancer : python "D:\Documents\A_conserver\Cours\Programmation\Labo IA\Akinator\akinator pour render\main.py"
