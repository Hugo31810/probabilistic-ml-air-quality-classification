#############################
## 1 PARTE - ENTRENAMIENTO ##
#############################

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline  # con esta librería vamos a crear nuestro pipeline de preprocesado
import pickle  # con esta librería nos guardaremos los objetos

# PREPROCESADO SOBRE LOS EJEMPLOS

# cargamos los ficheros de datos
print("Cargando X_train...")
X_train = pd.read_csv('practica_X_train.csv', sep=';') \
    .drop(columns=[0], errors='ignore') \
    .astype(float)
print("X_train shape:", X_train.shape)

print("\nCargando Y_train...")
Y_train = pd.read_csv('practica_Y_train.csv', sep=';')
Y_train.columns = ['Score', 'Air_Quality']
y_labels = Y_train['Air_Quality'].astype(str)
print("Primeras etiquetas (raw):", y_labels.head().tolist())

print("Ficheros de datos cargados con éxito...\n")

le = LabelEncoder()
y_enc = le.fit_transform(y_labels)
print("Primeros códigos:", y_enc[:5])
print("Mapping etiqueta:código:", dict(zip(le.classes_, le.transform(le.classes_))))

# creacion del pipeline de prepricesado
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # estandarizacion antes de las interacciones para no perder el contol de la magnitud de los términos
    ('poly', PolynomialFeatures(degree=2,
                                interaction_only=True,  # no tenemos interacciones cuadráticas
                                include_bias=False)),  # eliminar columna de unos
    ('scaler2', StandardScaler()),  # estandarizacion antes de pca
    ('pca', PCA(n_components=0.95).set_output(transform="pandas")),  # conserva el 95% de la varianza
])

print("Ajustando pipeline sobre x_train...\n")
pipeline.fit(X_train)

# nos guardamos los objetos para mas adelante
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# CREACION DEL MODELO A PRIORI DE LA ETIQUETA
class_count = Y_train['Air_Quality'].value_counts()  # e.g. Index(['Good','Moderate','Poor'], …)total = len(Y_train)
codes = le.transform(class_count.index)  # e.g. array([0,1,2])print("Modelo a priori:\n",apriori)
probs = (class_count / len(Y_train)).values  # e.g. array([0.43,0.35,0.22])
priors_dict = dict(zip(codes, probs))  # e.g. {0:0.43, 1:0.35, 2:0.22}

# supongo que opcional
# Si quieres guardarlo como dict para usarlo luego:
with open('priors.pkl', 'wb') as f:
    pickle.dump(priors_dict, f)

# CONSTRUCCION DE MODELOS DE VEROSIMILITUD
# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_labels)
# Guardamos el GNB
with open('model_gaussiannb.pkl', 'wb') as f:
    pickle.dump(gnb, f)

# MVN
clas_labels = np.unique(y_labels)
means = {}
covs = {}
for c in clas_labels:
    Xc = X_train[y_labels == c]
    means[c] = Xc.mean(axis=0)
    covs[c] = np.cov(Xc, rowvar=False, bias=True)

with open("model_mvn.pkl", "wb") as f:
    pickle.dump({'means': means, 'covs': covs}, f)

# GMM
from sklearn.mixture import GaussianMixture

n_components = 2
gmms = {}
for c in clas_labels:
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          random_state=0)
    gmm.fit(X_train[y_labels == c])
    gmms[c] = gmm

with open('model.gmm.pkl', 'wb') as f:
    pickle.dump({'gmms': gmm}, f)

# CONSTRUIR 3 SISTEMAS CLASIFICADORES
# GAUSSIAN NAIVE BAYES
clf_gnb = Pipeline([
    ('pre', pipeline),
    ('gnb', GaussianNB(priors=[priors_dict[c] for c in sorted(priors_dict)]))
])
# entrenamos el modelo
clf_gnb.fit(X_train, y_enc)
with open('clf_gnb.pkl', 'wb') as f:
    pickle.dump(clf_gnb, f)

# MVN completo
clf_mvn = Pipeline([
    ('pre', pipeline),
    ('qda', QuadraticDiscriminantAnalysis(
        priors=[priors_dict[c] for c in sorted(priors_dict)],
        store_covariance=True,
        reg_param=0.0  # puedes tunear regularización si Σ está cercana a singular
    ))
])
clf_mvn.fit(X_train, y_enc)
with open('clf_mvn.pkl', 'wb') as f:
    pickle.dump(clf_mvn, f)

# GMM por clase como clasificador Bayesiano
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin

# Parámetros según tu cuaderno de clase
n_components = 2
covariance_type = 'full'
max_iter = 100
random_state = 1460


# Cargamos el pipeline preajustado y los priors

class GMMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, preprocessor, priors,
                 n_components=2,
                 covariance_type='full',
                 max_iter=15,
                 random_state=None):
        self.preprocessor = preprocessor
        self.priors = priors
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        Xp = self.preprocessor.transform(X)  # 1) Transformamos X con el pipeline
        self.classes_ = np.unique(y)  # 2) Ajustamos un GMM por cada clase
        self.gmms_ = {}
        for c in self.classes_:
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                max_iter=self.max_iter,
                random_state=self.random_state
            )
            gmm.fit(Xp[y == c])
            self.gmms_[c] = gmm
        return self

    def predict(self, X):
        Xp = self.preprocessor.transform(X)
        # 3) Para cada clase, sumamos log-likelihood + log-prior y elegimos argmax
        logps = np.vstack([
            self.gmms_[c].score_samples(Xp) + np.log(self.priors[c])
            for c in self.classes_
        ]).T
        idx = np.argmax(logps, axis=1)
        return self.classes_[idx]


# Construcción y entrenamiento final
clf_gmm = GMMClassifier(
    preprocessor=pipeline,
    priors=priors_dict,
    n_components=n_components,
    covariance_type=covariance_type,
    max_iter=max_iter,
    random_state=random_state
)
clf_gmm.fit(X_train, y_enc)

# Serializamos el clasificador
with open('clf_gmm.pkl', 'wb') as f:
    pickle.dump(clf_gmm, f)

print("GMM classifier entrenado y guardado en 'clf_gmm.pkl'")

# COMPARACION DE RESULTADOS ENTRE MODELOS CON UN SUBCONJUTNO DE VALIDACION
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 4.1 Split  80/20
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=0
)

# Ajustar cada clasificador
clf_gnb.fit(X_tr, y_tr)
clf_mvn.fit(X_tr, y_tr)
clf_gmm.fit(X_tr, y_tr)

# Predecir en validación
y_pred_gnb = clf_gnb.predict(X_val)
y_pred_mvn = clf_mvn.predict(X_val)
y_pred_gmm = clf_gmm.predict(X_val)

# Métricas de accuracy
acc_gnb = accuracy_score(y_val, y_pred_gnb)
acc_mvn = accuracy_score(y_val, y_pred_mvn)
acc_gmm = accuracy_score(y_val, y_pred_gmm)

print("Accuracy GaussianNB:   ", acc_gnb)
print("Accuracy QDA (MVN):    ", acc_mvn)
print("Accuracy GMM:          ", acc_gmm, "\n")

# 4.5 Informe detallado por clase
print("=== Report GNB ===")
print(classification_report(y_val, y_pred_gnb, target_names=le.classes_))
print("=== Report QDA (MVN) ===")
print(classification_report(y_val, y_pred_mvn, target_names=le.classes_))
print("=== Report GMM ===")
print(classification_report(y_val, y_pred_gmm, target_names=le.classes_))




