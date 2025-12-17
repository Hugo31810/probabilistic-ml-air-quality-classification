import pandas as pd
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

#cargamos los objetos creados en el script de entrenamiento
with open(BASE_DIR / 'clf_mvn.pkl', 'rb') as f:
    clf = pickle.load(f)
with open(BASE_DIR / 'label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

X_test = pd.read_csv(BASE_DIR / 'practica_X_test.csv', sep=';')
X_test = X_test.drop(columns=[0], errors='ignore').astype(float)

#hacemos las predicciones
y_codes = clf.predict(X_test)

#descodificamos
y_labels = label_encoder.inverse_transform(y_codes)
ids = X_test.index + 1
output_df = pd.DataFrame(y_labels, columns=["Air_Quality"])

#guardamos con el mismo formato que practica_X_train.csv
with open(BASE_DIR / 'practica2_Y_test.csv', 'w', encoding='utf-8') as f:
    f.write(';Air_Quality\n')
    for index, label in zip(ids, y_labels):
        f.write(f"{index};{label}\n")

output_df.to_csv(BASE_DIR / 'practica2_Y_test.csv', index=True, sep=';')
