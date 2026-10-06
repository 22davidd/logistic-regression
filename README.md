<div align="center">

# 📈 logistic-regression

un mic cod facut de mine dupa un curs de la developers.google.com despre machine learning si LLM uri

![language](https://img.shields.io/badge/language-python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![dataset](https://img.shields.io/badge/date-30_seturi-2ea44f?style=for-the-badge)
![epochs](https://img.shields.io/badge/epochs-100.000-8957e5?style=for-the-badge)
![overfitting](https://img.shields.io/badge/overfitting-probabil-red?style=for-the-badge)

</div>

---

## 🎯 ce face
un model care prezice daca o persoana cumpara sau nu un produs, in functie de:
- **varsta**
- **salariu**

```mermaid
flowchart LR
    A["👤 varsta + salariu"] --> B["🧠 model"]
    B --> C["🛒 cumpara / nu cumpara"]
    C -. "x 100.000 repetari" .-> B
```

## 🏋️ antrenare

| | |
|---|---|
| 📥 **input** | varsta, salariu |
| 📤 **output** | cumpara / nu |
| 🗂️ **date** | 30 de seturi (varsta, salariu, a cumparat / nu) |
| 🔁 **epochs** | 100.000 |

## ⚠️ limitari
> [!WARNING]
> - 30 de exemple e foarte putin, deci **probabil face overfitting**
> - 100.000 de epochs pe un dataset atat de mic amplifica asta
> - este doar un mini proiect, nimic real

## 🚀 cum il rulezi? simplu
```bash
python3 logistic_regression.py
```

💖 pupici tuturor :* 💖

<div align="center">

</div>
