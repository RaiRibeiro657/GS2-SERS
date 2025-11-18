import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulação de dados
np.random.seed(42)
horas = np.arange(0, 24)
temperatura = 20 + 5 * np.sin(np.linspace(0, 3, 24))
ocupacao = np.random.choice([0, 1], size=24, p=[0.4, 0.6])

# Consumo base: maior com ocupação, menor à noite
consumo = 0.3 + ocupacao * 0.5 + np.random.normal(0, 0.05, 24)

df = pd.DataFrame({
    "Hora": horas,
    "Temperatura": temperatura.round(2),
    "Ocupação": ocupacao,
    "Consumo_kWh": consumo.round(3)
})

print(df)

# Identificação de desperdício: consumo alto sem ocupação
df["Desperdício"] = np.where((df["Ocupação"] == 0) & (df["Consumo_kWh"] > 0.35), 1, 0)

print("\nHoras com desperdício de energia:")
print(df[df["Desperdício"] == 1])

# Gráfico de consumo
plt.figure(figsize=(10, 5))
plt.plot(df["Hora"], df["Consumo_kWh"], marker='o')
plt.xlabel("Hora do dia")
plt.ylabel("Consumo (kWh)")
plt.title("Consumo Energético ao Longo do Dia")
plt.grid(True)
plt.show()

# Regressão linear simples (hora -> consumo)
model = LinearRegression()
model.fit(df[["Hora"]], df["Consumo_kWh"])
pred = model.predict(df[["Hora"]])

plt.figure(figsize=(10, 5))
plt.scatter(df["Hora"], df["Consumo_kWh"])
plt.plot(df["Hora"], pred)
plt.xlabel("Hora do dia")
plt.ylabel("Consumo (kWh)")
plt.title("Tendência do Consumo (Regressão Linear)")
plt.grid(True)
plt.show()
