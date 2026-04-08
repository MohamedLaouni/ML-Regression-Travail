#Version 1 — From Scratch


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Charger le datase
df = pd.read_csv('student_performance.csv')

# Vérification 1 — colonnes
print("=== Colonnes ===")
print(df.columns.tolist())

# Vérification 2 — premières lignes
print("\n=== Dimensions ===")
print(df.shape)










#------------------------------------------------------------
# 1) Régression Linéaire : On va prédire la note finale (Performance_Index) à partir des heures d'étude uniquement (Hours_Studied).


#choisir x et y a partir des colonnes de notre dataset
x=df['Hours_Studied'].values
y=df['Performance_Index'].values

# Etape 1 : calcule du moyenes
x_mean=np.mean(x)
y_mean=np.mean(y)

# Etape 2 : calcule de a et b a partire de la fonction de cout f(a,b)= (1/2n) · Σ ( yᵢ − ŷᵢ )²
# a = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
a= (np.sum((x-x_mean)*(y-y_mean)))/np.sum((x-x_mean)**2)

# b = ȳ − a · x̄
b=y_mean-a*x_mean

print("\n=== Résultats From Scratch ===")
print("Pente a :", round(a, 4))
print("Biais  b :", round(b, 4))
print("Formule  : y =", round(a,4), "* x +", round(b,4))

# Etap 3: Predire
y_pred= a*x + b

# Étape 4 : Calcule de la fonction de cout et R2
mse= np.mean((y-y_pred)**2)
R2=1-((np.sum((y-y_pred)**2))/(np.sum((y-y_mean)**2)))
print("\n=== Évaluation ===")
print("MSE :", round(mse, 2))
print("R²  :", round(R2,  4))

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, label='Données réelles')
plt.plot(x, y_pred, color='red', linewidth=2, label='Droite from scratch')
plt.xlabel('Heures d\'étude')
plt.ylabel('Note finale')
plt.title('Régression Linéaire Simple — From Scratch')
plt.legend()
plt.show()













#------------------------------------------------------------
#  2) Régression Linéaire Multiple :


# ── Choisir X et y ────────────────────────────────────────────────
X = df[['Hours_Studied', 'Previous_Score',
        'Sleep_Hours', 'Motivation', 'Absences']].values  # shape (30, 5)
y = df['Performance_Index'].values                         # shape (30,)

# ── Étape 1 : Ajouter la colonne de 1 pour le biais b ────────────
# La matrice X doit avoir une colonne de 1 au début
# Pourquoi ? Parce que b = b * 1  → le biais est le coefficient de 1
ones = np.ones((X.shape[0], 1))        # colonne de 1 → shape (30, 1)
X_b  = np.hstack([ones, X])            # coller les 1 à gauche → shape (30, 6)

print("Shape X_b :", X_b.shape)        # → (30, 6)

# ── Étape 2 : Formule matricielle θ = (XᵀX)⁻¹ · Xᵀy ─────────────
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

print("\n=== Coefficients From Scratch ===")
print("Biais        b  :", round(theta[0], 4))
print("Heures      a₁  :", round(theta[1], 4))
print("Note préc   a₂  :", round(theta[2], 4))
print("Sommeil     a₃  :", round(theta[3], 4))
print("Motivation  a₄  :", round(theta[4], 4))
print("Absences    a₅  :", round(theta[5], 4))

# ── Étape 3 : Prédire ─────────────────────────────────────────────
y_pred = X_b.dot(theta)       # shape (30,)

# ── Étape 4 : Évaluer ─────────────────────────────────────────────
y_mean = np.mean(y)
mse    = np.mean((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2     = 1 - (ss_res / ss_tot)

print("\n=== Évaluation ===")
print("MSE :", round(mse, 2))
print("R²  :", round(r2,  4))

# ── Étape 5 : Visualiser ──────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y)), y,      color='blue', label='Réel')
plt.scatter(range(len(y)), y_pred, color='red', label='Prédit')
plt.xlabel('Étudiant')
plt.ylabel('Note finale')
plt.title('Régression Linéaire Multiple — From Scratch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()





#------------------------------------------------------------
#  3) Régression Polynomiale From Scratch:



# On utilise une seule variable x (Hours_Studied)
# On crée des puissances : x, x², x³

x = df['Hours_Studied'].values
y = df['Performance_Index'].values

# ── Étape 1 : Créer la matrice polynomiale ────────────────────────
# degré 2 → colonnes : [1, x, x²]
# degré 3 → colonnes : [1, x, x², x³]
degre = 2

X_poly = np.column_stack([x**i for i in range(degre + 1)])
# range(3) → [0, 1, 2] → x⁰=1, x¹=x, x²=x²
# shape → (30, 3)

print("Shape X_poly :", X_poly.shape)   # → (30, 3)

# ── Étape 2 : Formule matricielle (même formule !) ────────────────
theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

print("\n=== Coefficients Polynomiaux ===")
print("b   (degré 0) :", round(theta[0], 4))
print("a₁  (degré 1) :", round(theta[1], 4))
print("a₂  (degré 2) :", round(theta[2], 4))

# ── Étape 3 : Prédire ─────────────────────────────────────────────
y_pred = X_poly.dot(theta)

# ── Étape 4 : Évaluer ─────────────────────────────────────────────
y_mean = np.mean(y)
mse    = np.mean((y - y_pred) ** 2)
ss_tot = np.sum((y - y_mean) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2     = 1 - (ss_res / ss_tot)

print("\n=== Évaluation ===")
print("MSE :", round(mse, 2))
print("R²  :", round(r2,  4))

# ── Étape 5 : Visualiser ──────────────────────────────────────────
x_line  = np.linspace(x.min(), x.max(), 100)
X_line  = np.column_stack([x_line**i for i in range(degre + 1)])
y_line  = X_line.dot(theta)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, label='Données réelles')
plt.plot(x_line, y_line, color='red', linewidth=2, label='Courbe polynomiale')
plt.xlabel('Heures d\'étude')
plt.ylabel('Note finale')
plt.title('Régression Polynomiale degré 2 — From Scratch')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
