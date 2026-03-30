#  Projet Deep Learning — CNN & LSTM

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/API-Flask-lightgrey?style=flat-square&logo=flask)
![CNN](https://img.shields.io/badge/CNN-81.39%25-brightgreen?style=flat-square)
![LSTM](https://img.shields.io/badge/LSTM-RMSE%200.5456-brightgreen?style=flat-square)

> Implémentation complète de deux modèles de Deep Learning :
> un **CNN** pour la classification d'images CIFAR-10
> et un **LSTM** pour la prédiction de séries temporelles météorologiques (Jena Climate).
> Le tout exposé via une **API Flask** avec interface web interactive 3 pages.

**Équipe :** Aelle · Nerge · Kiriane · Sandra · Leandra  
**Dépôt :** [github.com/aelle-prox/Projet-Deep-Learnig](https://github.com/aelle-prox/Projet-Deep-Learnig)

---

## Table des matières

1. [Présentation](#1--présentation)
2. [Résultats globaux](#2--résultats-globaux)
3. [Arborescence](#3--arborescence)
4. [Installation](#4--installation)
5. [Utilisation](#5--utilisation)
6. [Partie CNN — CIFAR-10](#6--partie-cnn--cifar-10)
7. [Partie LSTM — Jena Climate](#7--partie-lstm--jena-climate)
8. [API Flask](#8--api-flask)
9. [Commandes Git utiles](#9--commandes-git-utiles)
10. [Équipe](#10--équipe)

---

##  Présentation

Ce projet a pour objectif de passer de la **théorie mathématique** à une
**implémentation complète** en Deep Learning, en couvrant deux domaines
fondamentaux :

| Domaine | Modèle | Dataset | Objectif |
|---------|--------|---------|----------|
| 👁️ Vision par Ordinateur | CNN | CIFAR-10 | Accuracy > 70 % |
| 📈 Séries Temporelles | LSTM | Jena Climate | Minimiser RMSE & MAE |

Le projet suit une **architecture modulaire** conforme aux standards de
l'industrie : séparation claire entre chargement des données, preprocessing,
modèles, entraînement, évaluation et visualisation.

---

## 2 · Résultats globaux

| Modèle | Métrique | Valeur | Objectif | Statut |
|--------|----------|--------|----------|--------|
| **CNN** | Test Accuracy | **81.39 %** | > 70 % | 
| **CNN** | Test Loss | 0.5673 | 
| **CNN** | Meilleure époque | 33 / 50 | 
| **LSTM** | RMSE | **0.5456 °C** | Minimiser | 
| **LSTM** | MAE | **0.4355 °C** | Minimiser |
| **LSTM** | Meilleure époque | 23 | 

---

## 3 · Arborescence
```
Projet-Deep-Learning/
│
├── app.py                  ← API Flask (interface web + /api/predict)
├── train.py                ← Entraînement CNN
├── train_lstm.py           ← Entraînement LSTM
├── evaluate.py             ← Évaluation CNN
├── evaluate_lstm.py        ← Évaluation LSTM
├── requirements.txt        ← Dépendances Python
├── README.md
│
├── data/                   ← Datasets (CIFAR-10, Jena Climate CSV)
│   ├── jena_train.csv
│   ├── jena_val.csv
│   └── jena_test.csv
│
├── models/                 ← Architectures des modèles
│   ├── cnn_model.py        ← CustomCNN
│   └── rnn_model.py        ← LSTM
│
├── utils/                  ← Modules utilitaires
│   ├── data_loader.py      ← Chargement & pipeline CIFAR-10
│   ├── preprocessing.py    ← Normalisation MinMaxScaler
│   ├── windowing.py        ← Fenêtres temporelles (sliding window)
│   ├── visualization.py    ← Graphiques (loss, prédictions, split)
│   ├── metrics.py          ← Calcul RMSE et MAE
│   └── export.py           ← Export des résultats
│
├── outputs/                ← Graphiques générés
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── data_samples.png
│   ├── label_distribution.png
│   ├── augmented_images.png
│   └── split_verification.png
│
├── saved_model/            ← Modèles entraînés sauvegardés
│   └── best_cnn.keras
│
└── templates/              ← Pages HTML (interface Flask)
    ├── base.html
    ├── accueil.html
    ├── prediction.html
    └── metriques.html
```

---

##  · Installation

### Prérequis

- Python 3.9 ou supérieur
- pip
- Git

### Cloner le dépôt
```bash
git clone https://github.com/aelle-prox/Projet-Deep-Learnig.git
cd Projet-Deep-Learnig
```

### Créer un environnement virtuel
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Installer les dépendances
```bash
pip install -r requirements.txt
```

---

## 5 · Utilisation

### Entraînement
```bash
# Entraîner le CNN sur CIFAR-10
python train.py

# Entraîner le LSTM sur Jena Climate
python train_lstm.py
```

### Évaluation
```bash
# Évaluer le CNN
python evaluate.py

# Évaluer le LSTM
python evaluate_lstm.py
```

### Lancer l'interface web
```bash
python app.py
```

L'interface est disponible sur : **http://localhost:5000**

> Le dataset Jena Climate est téléchargé **automatiquement** au premier
> lancement via `keras.utils.get_file()`. Aucun téléchargement manuel requis.

---

## 6 · Partie CNN — CIFAR-10

### Dataset

| Paramètre | Valeur |
|-----------|--------|
| Nom | CIFAR-10 |
| Images | 60 000 (50 000 train · 10 000 test) |
| Dimensions | 32 × 32 pixels · 3 canaux RGB |
| Classes | 10 — équilibrées (5 000 images/classe en train) |

**Les 10 classes :**  
✈️ avion · 🚗 automobile · 🐦 oiseau · 🐱 chat · 🦌 cerf ·
🐶 chien · 🐸 grenouille · 🐴 cheval · ⛵ bateau · 🚚 camion

### Prétraitement
```python
# Normalisation pixels [0, 255] → [0.0, 1.0]
x_train = x_train.astype('float32') / 255.0

# Data Augmentation
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

### Architecture — CustomCNN
```
Input (32, 32, 3)
      ↓
Conv2D(64, 3×3) → BatchNorm → ReLU
Conv2D(64, 3×3) → BatchNorm → ReLU
MaxPooling2D → Dropout(0.3)
      ↓
Conv2D(128, 3×3) → BatchNorm → ReLU
Conv2D(128, 3×3) → BatchNorm → ReLU
MaxPooling2D → Dropout(0.4)
      ↓
Conv2D(256, 3×3) → BatchNorm → ReLU
MaxPooling2D → Dropout(0.5)
      ↓
Flatten → Dense(512) → Dropout(0.5)
      ↓
Dense(10, softmax)
```

### Compilation & callbacks
```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint(save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
]
```

### Résultats détaillés par classe

| Classe | Précision | Rappel | F1-score |
|--------|-----------|--------|----------|
| ✈️ avion | 84 % | 88 % | 86 % |
| 🚗 automobile | 88 % | 93 % | **90 %** |
| 🐦 oiseau | 79 % | 72 % | 75 % |
| 🐱 chat | 64 % | 63 % | 63 % |
| 🦌 cerf | 80 % | 77 % | 79 % |
| 🐶 chien | 72 % | 70 % | 71 % |
| 🐸 grenouille | 87 % | 89 % | **88 %** |
| 🐴 cheval | 85 % | 86 % | 86 % |
| ⛵ bateau | 93 % | 88 % | **90 %** |
| 🚚 camion | 84 % | 88 % | 86 % |

---

##  · Partie LSTM — Jena Climate

### Dataset

| Paramètre | Valeur |
|-----------|--------|
| Nom | Jena Climate Dataset |
| Source | Max Planck Institute for Biogeochemistry |
| Période | Janvier 2009 → Décembre 2016 |
| Fréquence brute | 1 mesure / 10 minutes |
| Fréquence utilisée | 1 mesure / heure |
| Lignes après nettoyage | 70 091 |

### Features retenues (5 variables)

| Feature | Description | Rôle |
|---------|-------------|------|
| `T (degC)` | Température de l'air | **Cible à prédire** |
| `p (mbar)` | Pression atmosphérique | Feature |
| `rh (%)` | Humidité relative | Feature |
| `wv (m/s)` | Vitesse du vent | Feature |
| `wd (deg)` | Direction du vent | Feature |

### Pipeline
```
Téléchargement automatique (keras.utils.get_file)
        ↓
Sous-échantillonnage [5::6]  →  10 min → 1 heure
        ↓
Sélection des 5 features
        ↓
Nettoyage  (wv clip ≥ 0 · dropna)
        ↓
Split chronologique strict  70 / 20
        ↓
Normalisation MinMaxScaler  (fit sur train uniquement)
        ↓
Sliding Window  →  séquences [h-N, ..., h-1]  →  T+1
        ↓
Entraînement LSTM
        ↓
Évaluation  →  RMSE · MAE
```

### Split chronologique 70 / 15 / 15

| Partition | Lignes | Période |
|-----------|--------|---------|
| **Train** | 49 063 | 2009-01-01 → 2014-08-05 |
| **Test** | 21 028|  2014-08-05 → 2016-12-31 |

> **Règle absolue** : le split est toujours chronologique, jamais aléatoire.
> Mélanger les données avant le split = data leakage = métriques faussées.

### Architecture — LSTM
```
Input (séquence temporelle, 5 features)
        ↓
LSTM(units, return_sequences=False)
        ↓
Dense(1)  ←  prédiction T+1 en °C
```
```python
model.compile(
    optimizer='adam',
    loss='mse'
)
```

### Résultats

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **RMSE** | **0.5456 °C** | Erreur quadratique moyenne |
| **MAE** | **0.4355 °C** | Écart moyen entre prédit et réel |
| Meilleure époque | 23 | Convergence rapide |

---

## API Flask

### Pages disponibles

| URL | Description |
|-----|-------------|
| `http://localhost:5000/` | Accueil — présentation & statistiques |
| `http://localhost:5000/prediction` | Prédiction CNN en temps réel |
| `http://localhost:5000/metriques` | Métriques & graphiques complets |

### Endpoint REST — `/api/predict`
```bash
curl -X POST http://localhost:5000/api/predict \
     -F "image=@ma_photo.jpg"
```
```json
{
  "label": "chien",
  "icon": "🐶",
  "confidence": 87.3,
  "probabilities": [
    { "label": "chien", "score": 87.3 },
    { "label": "chat",  "score":  5.1 }
  ],
  "mode": "model"
}
```


```



---




### Convention de commits recommandée

| Préfixe | Usage | Exemple |
|---------|-------|---------|
| `feat:` | Nouvelle fonctionnalité | `feat: add LSTM training script` |
| `fix:` | Correction de bug | `fix: correct data leakage in split` |
| `docs:` | Documentation | `docs: update README results` |
| `refactor:` | Refactorisation | `refactor: modularize data_loader` |

---

## Équipe

| Membre | Responsabilités |
|--------|----------------|
| **TCHATCHOUANG DJUICHI Aelle** | 
| **NOUMEDEM-MEGNIKENG Nergelo** | 
| **KUEKAM GOULAH Kiriane La Fortune** |
| **TCHAPGA TOUMI Nadège Sandra** | 
| **ONDOBO ENAMA Patricia Leandra** | 

---

