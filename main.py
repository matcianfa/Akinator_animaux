"""
API FastAPI pour Akinator
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
import csv
from pathlib import Path
import uuid

# Constantes (identiques à votre code)
NOM_CSV = "akinator_animaux.csv"
REPONSES_POSSIBLES = ["Oui", "Plutôt oui", "Je ne sais pas", "Plutôt non", "Non"]
VALEURS_REPONSES = [1, 0.75, 0.5, 0.25, 0]
SEUIL = 0.3

# Stockage des sessions en mémoire
sessions: Dict[str, dict] = {}

app = FastAPI(title="Akinator API")

# CORS pour permettre les appels depuis un frontend
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
    reponses_possibles: list

class AnswerRequest(BaseModel):
    session_id: str
    reponse: int  # 0-4 pour les 5 réponses possibles

class GuessResponse(BaseModel):
    animal: str
    probabilite: float
    is_final: bool
    question: Optional[str] = None
    question_number: Optional[int] = None

class ConfirmRequest(BaseModel):
    session_id: str
    correct: bool

# Fonctions de votre code (adaptées)
def donner_proba_animaux_sachant_r(r, donnees, proba_animaux):
    numerateurs = (1 - abs(r - donnees)) * proba_animaux
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
    if proba_animaux[i_premier] > proba_animaux[i_second] + SEUIL:
        return i_premier, proba_animaux[i_premier]
    else:
        return None, None

def actualisation_valeurs_theoriques(donnees, indice_animal, reponses_donnees, alpha=0.1):
    for question, reponse in reponses_donnees.items():
        donnees[question, indice_animal] += alpha * (VALEURS_REPONSES[reponse] - donnees[question, indice_animal])

def sauvegarde_csv(animaux, compteur_apparitions, questions, donnees):
    with open(NOM_CSV, 'w', newline='', encoding='utf-8') as fichier:
        writer = csv.writer(fichier)
        writer.writerow(["Question"] + animaux)
        writer.writerow(["Compteur d'apparitions"] + compteur_apparitions)
        for i in range(len(questions)):
            writer.writerow([questions[i]] + [f"{valeur:.3f}" for valeur in donnees[i]])

def charger_donnees():
    try:
        with open(NOM_CSV, 'r', encoding='utf-8') as fichier:
            reader = csv.reader(fichier)
            animaux = next(reader)[1:]
            compteur_apparitions = [int(val) for val in next(reader)[1:]]
            questions = []
            donnees = []
            for ligne in reader:
                questions.append(ligne[0])
                donnees.append([float(val) for val in ligne[1:]])
            donnees = np.asarray(donnees)
            return animaux, compteur_apparitions, questions, donnees
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Fichier '{NOM_CSV}' non trouvé")

# Routes API
@app.get("/")
def root():
    return {"message": "API Akinator - Utilisez /start pour commencer une session"}

@app.post("/start", response_model=SessionResponse)
def start_session():
    """Démarre une nouvelle session Akinator"""
    session_id = str(uuid.uuid4())

    # Charger les données
    animaux, compteur_apparitions, questions, donnees = charger_donnees()

    # Initialiser les probabilités
    apparitions_totales = sum(compteur_apparitions)
    proba_animaux = np.asarray([val / apparitions_totales for val in compteur_apparitions])

    # Créer la session
    sessions[session_id] = {
        "animaux": animaux,
        "compteur_apparitions": compteur_apparitions,
        "questions": questions,
        "donnees": donnees,
        "proba_animaux": proba_animaux,
        "reponses_donnees": {},
        "questions_pas_encore_posees": np.ones(len(questions), dtype=bool),
        "compteur_question": 0
    }

    # Choisir la première question
    i_question = choix_meilleure_question(donnees, proba_animaux, sessions[session_id]["questions_pas_encore_posees"])

    if i_question is None:
        raise HTTPException(status_code=500, detail="Aucune question disponible")

    sessions[session_id]["question_courante"] = i_question
    sessions[session_id]["compteur_question"] = 1

    return SessionResponse(
        session_id=session_id,
        question=questions[i_question],
        question_number=1,
        reponses_possibles=REPONSES_POSSIBLES
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

    # Enregistrer la réponse
    session["reponses_donnees"][i_question] = request.reponse
    session["questions_pas_encore_posees"][i_question] = False

    # Mettre à jour les probabilités
    session["proba_animaux"] = donner_proba_animaux_sachant_r(
        VALEURS_REPONSES[request.reponse],
        session["donnees"],
        session["proba_animaux"]
    )[0][i_question]

    # Vérifier si on a trouvé
    indice_meilleur_animal, proba = recherche_bonne_reponse(session["proba_animaux"], session["animaux"])

    if indice_meilleur_animal is not None:
        session["animal_propose"] = indice_meilleur_animal
        return GuessResponse(
            animal=session["animaux"][indice_meilleur_animal],
            probabilite=float(proba),
            is_final=True
        )

    # Choisir la prochaine question
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
        session["proba_animaux"][session["animal_propose"]] = 0
        del session["animal_propose"]

        # Continuer avec une nouvelle question
        i_question = choix_meilleure_question(
            session["donnees"],
            session["proba_animaux"],
            session["questions_pas_encore_posees"]
        )

        if i_question is None:
            i_max = np.argmax(session["proba_animaux"])
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
