# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:03:15 2025

@author: Labo-Axel
"""

import serial
import time
import csv

# Ouvrir le fichier CSV pour stocker les données
with open(r'C:\Users\Labo-Axel\Desktop\Justine\ThermosensorsDatas\25_03_07_3_Thermosensors1.csv', mode='a', newline='') as sensor_file :
    sensor_writer = csv.writer(sensor_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sensor_writer.writerow(["Time", "Device 1 Temperature (C)", "Device 2 Temperature (C)", "Device 3 Temperature (C)", "Device 4 Temperature (C)", "Device 5 Temperature (C)"])

# Configuration du port série
com = "COM9"  # Vérifier que c'est bien le bon sur l'Arduino
baud = 9600  # Même débit en bauds que sur l'Arduino

# Ouvrir la connexion série 
x = serial.Serial(com, baud, timeout=1)

# Attendre que l'Arduino soit prêt  
time.sleep(0.5)

count = 0
temperatures = []
while x.isOpen():
    # Lire une ligne de données envoyée par l'Arduino
    data = str(x.readline().decode('utf-8'))
    
    # Si des données ont été reçues
    if data:
        print("data :", data)  # Afficher les données dans la console
        
        # Vérifie si la ligne contient les températures
        if True:  # "Temp C:" in data:
            # Diviser la ligne pour extraire les informations de température
            values = data.split(" ")  # "Temp C: ")
            print("values :", values)
            
            # Test si len(values) == 9
            if len(values) == 9:
                if count == 0 or count < 6:
                    print("count1 :", count)
                    # Récupérer les températures des différents capteurs
                    temperatures.append(values[5])
                    print("values[5] :", values[5])
                    print("temperature[] :", temperatures)
                    count = count + 1
                
                # Ajouter un horodatage et enregistrer dans le CSV
                if count == 5:
                    with open(r'C:\Users\Labo-Axel\Desktop\Justine\ThermosensorsDatas\25_03_07_3_Thermosensors1.csv', mode='a', newline='') as sensor_file:
                        sensor_writer = csv.writer(sensor_file)
                        sensor_writer.writerow([time.asctime()] + temperatures)
                        temperatures = []
                        count = 0
                        print("FIN")

    # Pause pour éviter de surcharger le port série
    time.sleep(0.5)

# Fermer la connexion série
x.close()




#%%

import pandas as pd

# Specify the path to the CSV file
file_path = r'C:\Users\Labo-Axel\Desktop\Justine\ThermosensorsDatas\25_03_07_3_Thermosensors1.csv'

# Read the CSV file into a pandas DataFrame
try:
    df = pd.read_csv(file_path)
    print(df)  # This will print the entire DataFrame
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")



#%%

import matplotlib.pyplot as plt
import numpy as np

# Supposons que df soit votre DataFrame contenant les données
# Créez un nouvel axe temporel basé sur les spécifications
n = len(df)
#◘time_intervalometer = np.arange(0, n * 5, 5)  # Crée un tableau allant de 0 à (n-1)*5 secondes

plt.figure(figsize=(10, 6))
for i in range(1, 6):
    #plt.plot(time_intervalometer, df[f'Device {i} Temperature (C)'], '.', label=f'Device {i} Temperature (C)')
    x=df[f'Device {i} Temperature (C)'][1:]
    y = x.astype(np.float)
    plt.plot(df[f'Time'][1:],y, '.', label=f'Device {i} Temperature (C)')

plt.xlabel('Temps (s)')
plt.ylabel('Température (°C)')
plt.legend()
plt.title('Températures des 5 capteurs au fil du temps')

# Utiliser MaxNLocator pour limiter le nombre de ticks sur l'axe des x
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limite à 10 ticks maximum

# Rotation des labels pour une meilleure lisibilité
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
#%%

print(df)

#%%

import matplotlib.pyplot as plt
import numpy as np

# Supposons que df soit votre DataFrame contenant les données
# Créez un nouvel axe temporel basé sur les spécifications
n = len(df)
new_time = np.arange(0, n * 5, 5)  # Crée un tableau allant de 0 à (n-1)*5 secondes

# Déterminer les limites de l'axe y avec une marge
min_temp = df[[f'Device {i} Temperature (C)' for i in range(1, 6)]].min().min()
max_temp = df[[f'Device {i} Temperature (C)' for i in range(1, 6)]].max().max()
margin = 0.1 * (max_temp - min_temp)  # Marge de 10% de la plage de température

plt.figure(figsize=(10, 6))
for i in range(1, 6):
    plt.plot(new_time, df[f'Device {i} Temperature (C)'], '.', label=f'Device {i} Temperature (C)')

plt.xlabel('Temps (s)')
plt.ylabel('Température (°C)')
plt.ylim(min_temp - margin, max_temp + margin)  # Définir les limites de l'axe y avec une marge
plt.legend()
plt.title('Températures des 5 capteurs au fil du temps')

# Utiliser MaxNLocator pour limiter le nombre de ticks sur l'axe des x
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))  # Limite à 10 ticks maximum

# Rotation des labels pour une meilleure lisibilité
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import pandas as pd

# Supposons que df soit votre DataFrame
# Créer un DataFrame fictif de forme (63, 6) pour l'exemple


# Créer une nouvelle figure
plt.figure()

# Tracer chaque colonne séparément
plt.plot(df.iloc[:, 1], label='Colonne B')
#plt.plot(df.iloc[:, 2], label='Colonne C')
#plt.plot(df.iloc[:, 3], label='Colonne D')
#plt.plot(df.iloc[:, 5], label='Colonne F')

# Ajouter des labels et une légende
plt.xlabel('Index')
plt.ylabel('Valeurs')
plt.legend()
plt.title('Affichage des colonnes B, C, D et F')

# Afficher le graphique
plt.show()
