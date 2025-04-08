# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:37:48 2025

@author: Labo-Axel
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os

# Specify the path to the CSV file
file_path = '/Volumes/Expansion/Reseau/ThermosensorsDatas/25_02_26_Thermosensors5.csv'

# Read the CSV file into a pandas DataFrame

#%%
try:
    df = pd.read_csv(file_path)
    print(df)  # This will print the entire DataFrame
except FileNotFoundError:
    print(f"The file at {file_path} was not found.")
    
# Charge le fichier en utilisant le séparateur ';'
df = pd.read_csv(file_path, sep=';')


#%%

# Version pour tenter de trier les températures
'''
tab_initial_device1 = np.array([])
tab_initial_device2 = np.array([])
tab_initial_device3 = np.array([])
tab_initial_device4 = np.array([])
tab_initial_device5 = np.array([])

tab_initial_device1 = np.append(tab_initial_device1, df.iloc[0,5])
tab_initial_device2 = np.append(tab_initial_device2, df.iloc[0,6])
tab_initial_device3 = np.append(tab_initial_device3, df.iloc[0,7])
tab_initial_device4 = np.append(tab_initial_device4, df.iloc[0,8])
tab_initial_device5 = np.append(tab_initial_device5, df.iloc[0,9])
'''

import numpy as np

# Initialisation des tableaux de données et de temps
tab_trie_device1 = np.array([df.iloc[0, 5]])
tab_trie_device2 = np.array([df.iloc[0, 6]])
tab_trie_device3 = np.array([df.iloc[0, 7]])
tab_trie_device4 = np.array([df.iloc[0, 8]])
tab_trie_device5 = np.array([df.iloc[0, 9]])

time_device1 = np.array([df.iloc[0, 3]])  # Temps associé à tab_trie_device1
time_device2 = np.array([df.iloc[0, 3]])  # Temps associé à tab_trie_device2
time_device3 = np.array([df.iloc[0, 3]])  # Temps associé à tab_trie_device3
time_device4 = np.array([df.iloc[0, 3]])  # Temps associé à tab_trie_device4
time_device5 = np.array([df.iloc[0, 3]])  # Temps associé à tab_trie_device5

# Parcours des lignes et colonnes spécifiées
for i in range(1, 2000):  # De la ligne 1 à la ligne 797 (index 0 à 797)
    for j in range(5, 10):  # De la colonne 5 à la colonne 9 (index 5 à 9)
        val = df.iloc[i, j]
        time_val = df.iloc[i, 3]  # Récupération du temps associé à l'élément
        
        # Récupération des derniers éléments des tableaux
        last_val1 = tab_trie_device1[-1]
        last_val2 = tab_trie_device2[-1]
        last_val3 = tab_trie_device3[-1]
        last_val4 = tab_trie_device4[-1]
        last_val5 = tab_trie_device5[-1]
        
        # Calcul des différences absolues entre df.iloc[i, j] et les derniers éléments des tableaux
        diff1 = abs(val - last_val1)
        diff2 = abs(val - last_val2)
        diff3 = abs(val - last_val3)
        diff4 = abs(val - last_val4)
        diff5 = abs(val - last_val5)
        
        # Trouver le tableau avec la valeur la plus proche
        min_diff = min(diff1, diff2, diff3, diff4, diff5)
        
        # Ajouter la valeur au tableau avec la différence la plus petite
        if min_diff == diff1:
            tab_trie_device1 = np.append(tab_trie_device1, val)
            time_device1 = np.append(time_device1, time_val)  # Ajouter le temps associé
        elif min_diff == diff2:
            tab_trie_device2 = np.append(tab_trie_device2, val)
            time_device2 = np.append(time_device2, time_val)  # Ajouter le temps associé
        elif min_diff == diff3:
            tab_trie_device3 = np.append(tab_trie_device3, val)
            time_device3 = np.append(time_device3, time_val)  # Ajouter le temps associé
        elif min_diff == diff4:
            tab_trie_device4 = np.append(tab_trie_device4, val)
            time_device4 = np.append(time_device4, time_val)  # Ajouter le temps associé
        else:
            tab_trie_device5 = np.append(tab_trie_device5, val)
            time_device5 = np.append(time_device5, time_val)  # Ajouter le temps associé

#%%


import matplotlib.pyplot as plt


plt.figure()
plt.plot(time_device1, tab_trie_device1,'.', label='Device 1', color='blue')
#plt.plot(time_device2, tab_trie_device2,'.', label='Device 2', color='green')
#plt.plot(time_device3, tab_trie_device3,'.', label='Device 3', color='red')
#plt.plot(time_device4, tab_trie_device4,'.', label='Device 4', color='orange')
#plt.plot(time_device5, tab_trie_device5,'.', label='Device 5', color='purple')


plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Device Values vs Time')

plt.legend()
plt.show()


#%%


import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Fonction pour convertir un temps "hh:mm:ss" en secondes depuis minuit
def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second


# Convertir les tableaux de temps en secondes
time_device1_seconds = np.array([time_to_seconds(t) for t in time_device1])
time_device2_seconds = np.array([time_to_seconds(t) for t in time_device2])
time_device3_seconds = np.array([time_to_seconds(t) for t in time_device3])
time_device4_seconds = np.array([time_to_seconds(t) for t in time_device4])
time_device5_seconds = np.array([time_to_seconds(t) for t in time_device5])



time_device1_seconds -= time_device1_seconds[0]
time_device2_seconds -= time_device2_seconds[0]
time_device3_seconds -= time_device3_seconds[0]
time_device4_seconds -= time_device4_seconds[0]
time_device5_seconds -= time_device5_seconds[0]


plt.figure()

plt.plot(time_device1_seconds, tab_trie_device1,'.', label='Device 1', color='blue')
plt.plot(time_device2_seconds, tab_trie_device2,'.', label='Device 2', color='green')
plt.plot(time_device3_seconds, tab_trie_device3,'.', label='Device 3', color='red')
plt.plot(time_device4_seconds, tab_trie_device4,'.', label='Device 4', color='orange')
plt.plot(time_device5_seconds, tab_trie_device5,'.', label='Device 5', color='purple')

plt.ylabel('T°C')
plt.xlabel('Time (s)')
plt.title('Device temp vs Time 25_02_26_5')
plt.legend(prop={'size': 10})
plt.show()








#%%

#Version qui limite les étiquettes sur l'axe des abcisses 

'''
import matplotlib.pyplot as plt
import numpy as np


n = len(df)


plt.figure(figsize=(10, 6))
for i in range(1, 6): #Parce que on a 5 capteurs
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
'''




#%%


## TRAITEMENT SANS TRI DES DONNEES

from datetime import datetime

# Fonction pour convertir une heure au format hh:mm:ss en secondes depuis minuit
def time_to_seconds(time_str):
    # Convertit la chaîne de caractères en objet datetime
    time_obj = datetime.strptime(time_str, '%H:%M:%S')
    
    # Calcul des secondes depuis 00:00:00
    seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return seconds


# Liste des heures
# Télecharger les temps et les séparer par des virgules
series_to_string = result = ', '.join(df.iloc[:, 3])
times = [series_to_string]

# Séparer la chaîne par la virgule et supprimer les espaces en excès
times = times[0].split(', ')

# Convertir la première heure en temps 0
base_time = time_to_seconds(times[0])

# Calculer les intervalles en secondes par rapport à la première heure
intervals = [time_to_seconds(time) - base_time for time in times]

# Afficher les intervalles
print(intervals)


#%%

import matplotlib.pyplot as plt
import pandas as pd

#df le DataFrame

# Créer une nouvelle figure
#plt.figure()

# Tracer chaque colonne séparément
#plt.plot(intervals,df.iloc[:, 5], '.', label='Device 1')
#plt.plot(intervals,df.iloc[:, 6], '.', label='Device 2')
#plt.plot(intervals,df.iloc[:, 7], '.', label='Device 3')
#plt.plot(intervals,df.iloc[:, 8], '.', label='Device 4')
#plt.plot(intervals,df.iloc[:, 9], '.', label='Device 5')



#plt.plot(tab_trie_device1, '.', label='Device 1')
#plt.plot(tab_trie_device2, '.', label='Device 2')
#plt.plot(tab_trie_device3, '.', label='Device 3')
#plt.plot(tab_trie_device4, '.', label='Device 4')
#plt.plot(tab_trie_device5, '.', label='Device 5')






# Ajouter des labels et une légende
#plt.xlabel('Time')
#plt.ylabel('Temperature')
#plt.legend()
#plt.title('Thermo captors 1 to 5')

# Afficher le graphique
#plt.show()

#%%

# Enregistrement des données triées dans des fichiers texte

output_dir = "/Volumes/Expansion/Reseau/Manips/25_02_26_5/temperature_triees"

# Utiliser 
np.savetxt(f"{output_dir}/tab_trie_device1.txt", tab_trie_device1, delimiter=",", header="Device 1 Temperature", comments='')
np.savetxt(f"{output_dir}/tab_trie_device2.txt", tab_trie_device2, delimiter=",", header="Device 2 Temperature", comments='')
np.savetxt(f"{output_dir}/tab_trie_device3.txt", tab_trie_device3, delimiter=",", header="Device 3 Temperature", comments='')
np.savetxt(f"{output_dir}/tab_trie_device4.txt", tab_trie_device4, delimiter=",", header="Device 4 Temperature", comments='')
np.savetxt(f"{output_dir}/tab_trie_device5.txt", tab_trie_device5, delimiter=",", header="Device 5 Temperature", comments='')

#Utiliser fmt='%s' pour le format du temps
np.savetxt(f"{output_dir}/time_device1_seconds.txt", time_device1_seconds, fmt='%s', delimiter=",", header="Device 1 time", comments='')
np.savetxt(f"{output_dir}/time_device2_seconds.txt", time_device2_seconds, fmt='%s', delimiter=",", header="Device 2 time", comments='')
np.savetxt(f"{output_dir}/time_device3_seconds.txt", time_device3_seconds, fmt='%s', delimiter=",", header="Device 3 time", comments='')
np.savetxt(f"{output_dir}/time_device4_seconds.txt", time_device4_seconds, fmt='%s', delimiter=",", header="Device 4 time", comments='')
np.savetxt(f"{output_dir}/time_device5_seconds.txt", time_device5_seconds, fmt='%s', delimiter=",", header="Device 5 time", comments='')

#%%

# Pour que la tempétrature corresponde avec ma séquence d'image, il me faut la mesure de température toutes les 5sec

#Pas parfait parce que ça saute des multiples de 5 si on a une 

# A AMELIORER !


def find_indices_with_missing_multiples(time_device_seconds):
    indices = []
    prev_multiple = None

    for i in range(len(time_device_seconds)):
        t = time_device_seconds[i]

        # Si on trouve un multiple de 5
        if t % 5 == 0:
            if prev_multiple is None:
                indices.append(i)
            else:
                diff = t - prev_multiple
                if diff == 5:
                    indices.append(i)
                elif diff > 5:
                    # Il manque un ou plusieurs multiples de 5
                    # On cherche la première valeur > prev_multiple qui n'est pas un multiple mais proche
                    # Par exemple si prev_multiple = 55 et t = 65, on cherche s'il y a une valeur proche de 60
                    missing = np.arange(prev_multiple + 5, t, 5)  # Liste des manquants : [60] dans l'exemple
                    for m in missing:
                        # On cherche si la valeur m+1 est dans le tableau
                        idx_next = np.where(time_device_seconds == m + 1)[0]
                        if len(idx_next) > 0:
                            indices.append(idx_next[0])
            prev_multiple = t

    return np.array(indices)




# Application aux différents jeux de données
indices_device1 = find_indices_with_missing_multiples(time_device1_seconds)

temp_device1_at_indices = tab_trie_device1[indices_device1]

np.savetxt(f"{output_dir}/temp_device1_at_indices.txt", temp_at_indices, delimiter=",", header="temp_device1_at_indices", comments='')


# Affichage
print("Indices pour Device 1:", indices_device1)





