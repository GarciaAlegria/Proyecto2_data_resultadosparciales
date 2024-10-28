import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar el dataset
data = pd.read_csv(r'C:\Users\manue\OneDrive\Escritorio\Data_Science\Proyecto2\train.csv')
texts = data['discourse_text']  # Columna con el texto del discurso
labels = data['discourse_effectiveness']  # Columna con las etiquetas de efectividad

# Preprocesamiento de texto
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

data['cleaned_text'] = texts.apply(preprocess_text)

# Añadir características de longitud de texto
data['text_length'] = data['cleaned_text'].apply(len)
data['word_count'] = data['cleaned_text'].apply(lambda x: len(x.split()))

# Generación de embeddings usando TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
text_embeddings = vectorizer.fit_transform(data['cleaned_text']).toarray()

# Combinación de embeddings y características de longitud
additional_features = data[['text_length', 'word_count']].values
embeddings = np.hstack((text_embeddings, additional_features))

# Codificar las etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Separación de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels_encoded, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear las clases en el conjunto de entrenamiento
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Definir un espacio reducido para GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# Configurar y entrenar el modelo con GridSearchCV
search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
search.fit(X_train_res, y_train_res)
best_classifier = search.best_estimator_

# Predicción y evaluación
y_pred = best_classifier.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Graficar la distribución de etiquetas en el conjunto de datos
plt.figure(figsize=(8, 6))
sns.countplot(x=labels, order=label_encoder.classes_)
plt.title("Distribución de la Efectividad del Discurso en el Dataset")
plt.xlabel("Efectividad")
plt.ylabel("Frecuencia")
plt.show()

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()