from flask import Flask, render_template, request, redirect, url_for ,jsonify
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px  # Assurez-vous que plotly est importé correctement
import io
import sys
import base64
import seaborn as sns


app = Flask(__name__)

# Charger les données
DATA_PATH = 'students_adaptability_level_online_education.csv'
data = pd.read_csv(DATA_PATH)

@app.route('/')
def base():
    return render_template('base.html', title="Accueil")
@app.route('/students', methods=['GET', 'POST'])
def students():
    query = request.args.get('search', '')
    sort_by = request.args.get('sort_by', 'Age')
    
    # Filtrer les étudiants
    filtered_data = data[data.apply(lambda row: query.lower() in row.to_string().lower(), axis=1)] if query else data
    
    # Trier les données
    if sort_by in data.columns:
        filtered_data = filtered_data.sort_values(by=sort_by)
    
    table = filtered_data.to_html(classes='table table-striped', index=False)
    return render_template('students.html', table=table, query=query, sort_by=sort_by)


@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    global data  # Déclarez d'abord la variable comme globale
    if request.method == 'POST':
        # Ajouter un nouvel étudiant
        new_student = {
            col: request.form[col] for col in data.columns if col in request.form
        }
        data = data.append(new_student, ignore_index=True)
        data.to_csv(DATA_PATH, index=False)
        return redirect(url_for('students'))
    return render_template('add_student.html', columns=data.columns)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = {col: request.form[col] for col in ['Age', 'Gender', 'Education Level', 
                                                         'IT Student', 'Location', 'Load-shedding', 
                                                         'Financial Condition', 'Internet Type', 
                                                         'Network Type', 'Class Duration', 
                                                         'Self Lms', 'Device']}
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
        prediction = model.predict(input_encoded)[0]
        predictions_history.append(input_data | {'Prediction': prediction})
        return render_template('predict.html', prediction=prediction)
    return render_template('predict_form.html', columns=data.columns)

predictions_history = []

@app.route('/predictions_history')
def predictions_history_page():
    return render_template('predictions_history.html', predictions=predictions_history)

@app.route('/data_info')
def data_info():
    # Résumé des données sans afficher la ligne de type
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue().split('\n')[1:]
    info_str = "\n".join(info_str)
    
    # Statistiques descriptives
    describe = data.describe()
    
    # Valeurs manquantes
    missing_values = data.isnull().sum()

    return render_template('data_info.html', 
                           data_info=info_str,
                           describe=describe.to_html(classes='table table-striped'),
                           missing_values=missing_values.to_frame().to_html(classes='table table-striped'))

from sklearn.metrics import classification_report

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    # Vérification des colonnes attendues
    expected_columns = ['Age', 'Gender', 'Education Level', 'IT Student', 'Location', 
                        'Load-shedding', 'Financial Condition', 'Internet Type', 
                        'Network Type', 'Class Duration', 'Self Lms', 'Device']
    missing_columns = [col for col in expected_columns if col not in data.columns]
    
    if missing_columns:
        return f"Missing columns: {', '.join(missing_columns)}", 400

    # Préparer les données
    features = expected_columns
    target = 'Adaptivity Level'
    
    X = data[features]
    y = data[target]
    
    # Encodage des variables catégorielles
    X_encoded = pd.get_dummies(X)

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Entraîner le modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    classification_metrics = classification_report(y_test, y_pred, output_dict=True)

    # Extraction des métriques principales
    precision = classification_metrics['weighted avg']['precision']
    recall = classification_metrics['weighted avg']['recall']
    f1_score = classification_metrics['weighted avg']['f1-score']

    # Génération de la matrice de confusion
    confusion = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=ax)
    plt.title('Confusion Matrix')
    plt.xlabel('Predictions')
    plt.ylabel('Actual Values')

    # Sauvegarder la matrice de confusion
    graph_path = os.path.join('static', 'confusion_matrix.png')
    plt.savefig(graph_path)
    plt.close()

    # Rendu des métriques et du graphique dans le template
    return render_template(
        'train_model.html',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        graph_path=graph_path
    )


############## GRAPHIQUES ################
@app.route('/adaptivity_distribution')
def adaptivity_distribution():
    plt.figure(figsize=(8, 6))
    data['Adaptivity Level'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Distribution des niveaux d'adaptabilité")
    plt.xlabel("Adaptability Level")
    plt.ylabel("Number of Students")

    graph_path = os.path.join('static', 'adaptivity_distribution.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('adaptivity.html', graph_path=graph_path)

@app.route('/gender_distribution')
def gender_distribution():
    plt.figure(figsize=(6, 6))
    data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title("Répartition par genre")

    graph_path = os.path.join('static', 'gender_distribution.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('gender.html', graph_path=graph_path)

@app.route('/education_level_distribution')
def education_level_distribution():
    plt.figure(figsize=(8, 6))
    data['Education Level'].value_counts().plot(kind='bar', color='orange')
    plt.title("Répartition par niveau d'éducation")
    plt.xlabel("Education Level")
    plt.ylabel("Number of students")

    graph_path = os.path.join('static', 'education_level_distribution.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('education.html', graph_path=graph_path)

@app.route('/age_vs_adaptivity')
def age_vs_adaptivity():
    plt.figure(figsize=(8, 6))
    data.plot.scatter(x='Age', y='Adaptivity Level', alpha=0.7, color='purple')
    plt.title("Relation entre âge et niveau d'adaptabilité")
    plt.xlabel("Age")
    plt.ylabel("Adaptability Level")

    graph_path = os.path.join('static', 'age_vs_adaptivity.png')
    plt.savefig(graph_path)
    plt.close()

    return render_template('agevsadap.html', graph_path=graph_path)
@app.route('/graphique')
def graphique():
    # Chemin pour enregistrer les graphiques
    graph_folder = 'static/graphs'
    if not os.path.exists(graph_folder):
        os.makedirs(graph_folder)

    # Graphique 1 : Distribution des niveaux d'adaptabilité
    plt.figure(figsize=(8, 6))
    data['Adaptivity Level'].value_counts().plot(kind='bar', color='skyblue')
    plt.title("Distribution of adaptability levels")
    plt.xlabel("Level of adaptability")
    plt.ylabel("Number of students")
    graph1_path = os.path.join(graph_folder, 'adaptivity_distribution.png')
    plt.savefig(graph1_path)
    plt.close()

    # Graphique 2 : Répartition par genre
    plt.figure(figsize=(8, 6))
    data['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'])
    plt.title("Répartition par genre")
    graph2_path = os.path.join(graph_folder, 'gender_distribution.png')
    plt.savefig(graph2_path)
    plt.close()

    # Graphique 3 : Répartition par niveau d'éducation
    plt.figure(figsize=(8, 6))
    data['Education Level'].value_counts().plot(kind='bar', color='orange')
    plt.title("Répartition par niveau d'éducation")
    plt.xlabel("Education Level")
    plt.ylabel("Number of students")
    graph3_path = os.path.join(graph_folder, 'education_level_distribution.png')
    plt.savefig(graph3_path)
    plt.close()

    # Graphique 4 : Relation entre âge et adaptabilité
    plt.figure(figsize=(8, 6))
    data.plot.scatter(x='Age', y='Adaptivity Level', alpha=0.7, color='purple')
    plt.title("Relation entre âge et niveau d'adaptabilité")
    plt.xlabel("Age")
    plt.ylabel("Adaptability Level")
    graph4_path = os.path.join(graph_folder, 'age_vs_adaptivity.png')
    plt.savefig(graph4_path)
    plt.close()

    # Envoyer les chemins des graphiques à la page HTML
    graphs = [graph1_path, graph2_path, graph3_path, graph4_path]
    return render_template('graph.html', graphs=graphs)

@app.route('/compare_models')
def compare_models():
    from sklearn.linear_model import LogisticRegression
    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)
    return render_template('compare_models.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
