#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Cargar librerias importantes para el Analisis de los archivos csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from scipy import stats


# # Limpieza y adquisicion de datos

# In[16]:


#Cargar el archivo en un dataframe para su manipulacion
df = pd.read_csv(r"C:\Users\marce\OneDrive\Desktop\Python\articulos-python\TESIS\NOM035.csv", encoding='ISO-8859-1')
display(df)


# In[18]:




# Display the data types of each column
print(df.dtypes)


# # Analisis Exploratorio de datos NOM-035 GRIII Resultados de 44 empleados

# In[7]:


df.describe()


# In[4]:


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.countplot(x="Nivel de Riesgo", data=df, palette="coolwarm", order=["Nulo", "Bajo", "Medio", "Alto", "Muy alto"])
ax.set_title("Distribución de Niveles de Riesgo", fontsize=18)
ax.set_xlabel("Nivel de Riesgo", fontsize=14)
ax.set_ylabel("Cantidad de Participantes", fontsize=14)
plt.show()


# In[6]:


plt.figure(figsize=(10, 6))
ax = sns.countplot(x="Nivel de Riesgo", hue="Género", data=df, palette="coolwarm", order=["Nulo", "Bajo", "Medio", "Alto", "Muy alto"])
ax.set_title("Distribución de Niveles de Riesgo por Género", fontsize=18)
ax.set_xlabel("Nivel de Riesgo", fontsize=14)
ax.set_ylabel("Cantidad de Participantes", fontsize=14)
plt.legend(title="Género")
plt.show()


# In[7]:


plt.figure(figsize=(12, 6))
ax = sns.countplot(x="Puesto de trabajo", hue="Nivel de Riesgo", data=df, palette="coolwarm")
ax.set_title("Distribución de Niveles de Riesgo por Puesto de Trabajo", fontsize=18)
ax.set_xlabel("Puesto de Trabajo", fontsize=14)
ax.set_ylabel("Cantidad de Participantes", fontsize=14)
plt.legend(title="Nivel de Riesgo")
plt.show()


# In[8]:


def color_risk(value):
    if value == "Alto":
        return 'background-color: red'
    elif value == "Medio":
        return 'background-color: yellow'
    elif value == "Bajo":
        return 'background-color: green'
    else:
        return ''

def risk_text(value):
    if value == "Alto":
        return "Se requiere realizar un análisis de cada categoría y dominio, de manera que se puedan determinar las acciones de intervención apropiadas a través de un Programa de intervención, que podrá incluir una evaluación específica1 y deberá incluir una campaña de sensibilización, revisar la política de prevención de riesgos psicosociales y programas para la prevención de los factores de riesgo psicosocial, la promoción de un entorno organizacional favorable y la prevención de la violencia laboral, así como reforzar su aplicación y difusión."
    elif value == "Medio":
        return "Se requiere revisar la política de prevención de riesgos psicosociales y programas para la prevención de los factores de riesgo psicosocial, la promoción de un entorno organizacional favorable y la prevención de la violencia laboral, así como reforzar su aplicación y difusión, mediante un Programa de intervención."
    elif value == "Bajo":
        return "Es necesario una mayor difusión de la política de prevención de riesgos psicosociales y programas para: la prevención de los factores de riesgo psicosocial, la promoción de un entorno organizacional favorable y la prevención de la violencia laboral."
    else:
        return ""


# In[ ]:


df['Acciones'] = df['Nivel de Riesgo'].apply(risk_text)

styled_df = df.style.applymap(color_risk, subset=['Nivel de Riesgo']).hide_index()
display(styled_df)


# # Extraccion de caracteristicas para Modelo
# 
# 

# In[20]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\marce\OneDrive\Desktop\Python\articulos-python\TESIS\NOM035.csv", encoding='ISO-8859-1')

# Select the object columns to encode and create a new DataFrame with those columns
object_columns = df.select_dtypes(include=['object']).columns
df_encoded = df[object_columns].copy()

# Perform one-hot encoding using pandas get_dummies function
df_encoded = pd.get_dummies(df_encoded)

# Compute the correlation matrix
corr_matrix = df_encoded.corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)

# Plot the correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"C:\Users\marce\OneDrive\Desktop\Python\articulos-python\TESIS\NOM035.csv", encoding='ISO-8859-1')

# Select the columns to encode and create a new DataFrame with those columns
columns_to_encode = ["Género", "Puesto de trabajo", "Nivel de riesgo"]
df_encoded = df[columns_to_encode].copy()

# Perform one-hot encoding using pandas get_dummies function
df_encoded = pd.get_dummies(df_encoded)

# Compute the correlation matrix
corr_matrix = df_encoded.corr()

# Plot the correlation heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# # Procesamiento,Extraccion y analisis de datos EEG de 30 pacientes

# In[5]:


import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load the CSV data
filepath = "C:\\Users\\marce\\OneDrive\\Desktop\\Python\\articulos-python\\TESIS\\task_relaxation_pacientes\\clean_1paciente.csv"
df = pd.read_csv(filepath)

# Select only Fp1, Fp2, F3, and F4 columns
eeg_data = df[['Fp1', 'Fp2', 'F3', 'F4']]

# Define function for artifact removal and filtering
def preprocess_eeg(eeg_data, lowcut=0.5, highcut=30, fs=128, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.lfilter(b, a, eeg_data)
    return filtered_data

# Clean the signal
filtered_eeg = preprocess_eeg(eeg_data)

# Define function for Welch's method
def welch_psd(filtered_eeg, fs=60, nperseg=4, noverlap=0):
    freqs, psd = signal.welch(filtered_eeg, fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, psd




# Compute PSD using Welch's method
freqs, psd = welch_psd(filtered_eeg)

# Define function to compute power ratios
def compute_power_ratios(freqs, psd):
    alpha_idx = np.where((freqs >= 8) & (freqs <= 13))
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))

    alpha_power = np.sum(psd[:, alpha_idx], axis=-1)
    beta_power = np.sum(psd[:, beta_idx], axis=-1)

    alpha_beta_ratio = alpha_power / beta_power
    return alpha_power, beta_power, alpha_beta_ratio

alpha_power, beta_power, alpha_beta_ratio = compute_power_ratios(freqs, psd)

#print("Alpha power:", alpha_power)
#print("Beta power:", beta_power)



# Calculate Alpha/Beta ratio
alpha_beta_ratio = compute_power_ratios(freqs, psd)
print("alpha_beta_ratio")

# Define function to plot Alpha/Beta ratio before and after stimulus
def plot_alpha_beta_ratio(alpha_beta_ratio, stimulus_time=13*60+55, fs=128, window=5):
    pre_stimulus_start = stimulus_time - window
    pre_stimulus_end = stimulus_time
    post_stimulus_start = stimulus_time
    post_stimulus_end = stimulus_time + window

    pre_stimulus_ratio = np.mean(alpha_beta_ratio[:, pre_stimulus_start*fs:pre_stimulus_end*fs], axis=-1)
    post_stimulus_ratio = np.mean(alpha_beta_ratio[:, post_stimulus_start*fs:post_stimulus_end*fs], axis=-1)
    
    #print("Alpha power:", alpha_power)
    #print("Beta power:", beta_power)

    plt.bar(['Before Stimulus', 'After Stimulus'], [pre_stimulus_ratio, post_stimulus_ratio])
    plt.ylabel('Alpha/Beta Ratio')
    plt.show()

