import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos desde un archivo CSV
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")

# Visualizar los datos
sns.scatterplot(x='Celsius', y='Fahrenheit', data=temperature_df)
plt.title("Celsius vs Fahrenheit")
plt.xlabel("Celsius")
plt.ylabel("Fahrenheit")
plt.show()

# Preparar los datos para el modelo
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1]) # 1 Neurona
])

# model.summary()

# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(1.0), loss='mean_squared_error')

# Entrenar el modelo
epochs_history = model.fit(x_train, y_train, epochs=100)

# Visualizar el progreso del entrenamiento
epochs_history.history.keys()

plt.plot(epochs_history.history['loss'])
plt.title('Progreso del entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Pérdida')
plt.show()

model.get_weights()

# Hacer predicciones
test_celsius = np.array([-40, 0, 37, 100])
predicted_fahrenheit = model.predict(test_celsius)
for i, celsius in enumerate(test_celsius):
    print(f"{celsius} grados Celsius son aproximadamente {predicted_fahrenheit[i][0]:.2f} grados Fahrenheit")
    
# Guardar el modelo entrenado
# model.save("celsius_to_fahrenheit_model.h5")
 