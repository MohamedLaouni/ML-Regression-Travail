#Version 2 — Avec scikit-learn

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Charger le datase
df = pd.read_csv('student_performance.csv')




#-------------------------------------------------
# 1) Régression Linéaire : On va prédire la note finale (Performance_Index) à partir des heures d'étude uniquement (Hours_Studied).
#choisir x et y a partir des colonnes de notre dataset
x=df['Hours_Studied'].values.reshape(-1,1)
y=df['Performance_Index'].values

print(x)
print(y)
# Entraîner le modèle
model = LinearRegression()
model.fit(x, y)

# Récupérer a et b :
print("Pente a :", round(model.coef_[0], 4))
print("Biais  b :", round(model.intercept_, 4))

# Prédire
y_pred = model.predict(x)

# Calcule de la fonction de cout et R2
mse = mean_squared_error(y, y_pred)
r2  = r2_score(y, y_pred)
print("MSE :", round(mse, 2))
print("R²  :", round(r2,  4))







#------------------------------------------------------------
#  2) Régression Linéaire Multiple avec Scikit learn :


# ── Choisir X et y ────────────────────────────────────────────────
# Pas besoin d'ajouter les 1 → sklearn le fait automatiquement !
X = df[['Hours_Studied', 'Previous_Score',
        'Sleep_Hours', 'Motivation', 'Absences']].values  # shape (30, 5)
y = df['Performance_Index'].values

# ── Entraîner ─────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X, y)

# ── Coefficients ──────────────────────────────────────────────────
print("=== Coefficients sklearn ===")
print("Biais        b  :", round(model.intercept_, 4))
print("Heures      a₁  :", round(model.coef_[0],   4))
print("Note préc   a₂  :", round(model.coef_[1],   4))
print("Sommeil     a₃  :", round(model.coef_[2],   4))
print("Motivation  a₄  :", round(model.coef_[3],   4))
print("Absences    a₅  :", round(model.coef_[4],   4))

# ── Prédire ───────────────────────────────────────────────────────
y_pred = model.predict(X)

# ── Évaluer ───────────────────────────────────────────────────────
mse = mean_squared_error(y, y_pred)
r2  = r2_score(y, y_pred)

print("\n=== Évaluation ===")
print("MSE :", round(mse, 2))
print("R²  :", round(r2,  4))

# ── Visualiser ────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.scatter(range(len(y)), y,      color='blue', alpha=0.6, label='Réel')
plt.scatter(range(len(y)), y_pred, color='red',  alpha=0.6, label='Prédit')
plt.xlabel('Étudiant')
plt.ylabel('Note finale')
plt.title('Régression Linéaire Multiple — scikit-learn')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()






#------------------------------------------------------------
#  3) Régression Polynomiale avec scikit learn :

from sklearn.preprocessing import PolynomialFeatures

# ── Choisir x et y ────────────────────────────────────────────────
x = df['Hours_Studied'].values.reshape(-1, 1)
y = df['Performance_Index'].values

# ── Étape 1 : Créer les features polynomiales ─────────────────────
# PolynomialFeatures fait exactement ce qu'on a fait from scratch
poly   = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)
# fit_transform ajoute automatiquement [1, x, x²]

print("Shape X_poly :", X_poly.shape)   # → (30, 3)

# ── Étape 2 : Entraîner ───────────────────────────────────────────
model = LinearRegression()
model.fit(X_poly, y)

# ── Coefficients ──────────────────────────────────────────────────
print("\n=== Coefficients sklearn ===")
print("b   (degré 0) :", round(model.intercept_,  4))
print("a₁  (degré 1) :", round(model.coef_[1],    4))
print("a₂  (degré 2) :", round(model.coef_[2],    4))

# ── Prédire ───────────────────────────────────────────────────────
y_pred = model.predict(X_poly)

# ── Évaluer ───────────────────────────────────────────────────────
mse = mean_squared_error(y, y_pred)
r2  = r2_score(y, y_pred)

print("\n=== Évaluation ===")
print("MSE :", round(mse, 2))
print("R²  :", round(r2,  4))

# ── Visualiser ────────────────────────────────────────────────────
import numpy as np
x_line  = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
X_line  = poly.transform(x_line)
y_line  = model.predict(X_line)

plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', alpha=0.6, label='Données réelles')
plt.plot(x_line, y_line, color='red', linewidth=2, label='Courbe polynomiale')
plt.xlabel('Heures d\'étude')
plt.ylabel('Note finale')
plt.title('Régression Polynomiale degré 2 — scikit-learn')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()