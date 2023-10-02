# Importamos las librerías necesarias
import dash
from dash import dcc 
from dash import html
import plotly.express as px
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from pgmpy.inference import VariableElimination
from sklearn.metrics import f1_score
from ucimlrepo import fetch_ucirepo
from dash.dependencies import Input, Output


# Creamos la red
 # fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

data=pd.DataFrame.from_dict(predict_students_dropout_and_academic_success['data']['original'])
data
#Tratamiento de los datos

#Eliminar todas las observaciones donde el Target sea 'Enrolled'
data_mod = data[data['Target'] != 'Enrolled']

#Categorizar variable Age of Enrollment
bins = [0, 19, 21, 24, 31, 41, float('inf')]  # Límites de los rangos
labels = ['Rango 16-18', 'Rango 19-20', 'Rango 21-23', 'Rango 24-30', 'Rango 31-40', 'Rango >41'] 
data_mod['age_range'] = pd.cut(data_mod['Age at enrollment'], bins=bins, labels=labels, right=False)

#Categorizar variable Curricular units 1st sem (approved)
bins = [0, 3, 6, 11, float('inf')]  # Límites de los rangos
labels = ['Rango 0-2', 'Rango 3-5', 'Rango 6-10', 'Rango >10'] 
data_mod['approved_sem1_range'] = pd.cut(data_mod['Curricular units 1st sem (approved)'], bins=bins, labels=labels, right=False)

#Categorizar variable Curricular units 1st sem (grade)
bins = [0, 10,11, 12,13, 14,16,float('inf')]  # Límites de los rangos
labels = ['Rango 0-10', 'Rango 10-11','Rango 11-12', 'Rango 12-13', 'Rango 13-14','Rango 14-16', 'Rango >16'] 
data_mod['grade_sem1_range'] = pd.cut(data_mod['Curricular units 1st sem (grade)'], bins=bins, labels=labels, right=False)

#Categorizar variable Admission grade
bins = [0, 100,115, 130,145, 160,float('inf')]  # Límites de los rangos
labels = ['Rango 0-100', 'Rango 100-115','Rango 115-130', 'Rango 130-145', 'Rango 145-160', 'Rango >160'] 
data_mod['grade_admission_range'] = pd.cut(data_mod['Admission grade'], bins=bins, labels=labels, right=False)

# Verifica el resultado
data_mod
# Definir una semilla
random.seed(12232)

# Definir el modelo
model = BayesianNetwork([("Gender", "Target"),("Displaced","Target"),
                         ("Scholarship holder","Target"),("Debtor","Target"),("Tuition fees up to date","Target"),
                         ("age_range","Target"),("approved_sem1_range","Target"),("grade_admission_range","approved_sem1_range")])

variables_seleccionadas = ['Gender', 'Displaced', 'Scholarship holder', 'Debtor',
                           'Tuition fees up to date','age_range','approved_sem1_range','grade_admission_range','Target']
# Nos quedamos con el dataset únicamente de las variables seleccionadas
data_sel = data_mod[variables_seleccionadas]
data_sel
train, test = train_test_split(data_sel, test_size=0.2, random_state=98)
#Calcular los parametros> probabilidades condicionales mediante maxima verosimilitud

emv = MaximumLikelihoodEstimator(model=model, data=train)

model.fit(data=train, estimator = MaximumLikelihoodEstimator)

#Chequiar modelo
model.check_model()
infer = VariableElimination(model)


data = {
    'nombre': ['Juan Pablo Ríos Hernández', 'Samuel Felipe Ríos Parra', 'Felipe Sanabria Trimiño', 'Andrés Felipe Sanabria Cotrino', 'Mayerli Andrea Velandia León','María Gabriela Urango Llorente'],
    'Codigo':[201821819, 201821820,201821926,201815259,201817500,202013781],
    'Gender':[1,1,1,1,0,0],
    'Displaced':[0,0,1,0,0,1],
    'Scholarship holder':[0,0,0,0,0,1],
    'Debtor':[0,0,0,0,0,0],
    'Tuition fees up to date':[1,1,1,1,1,1],
    'age_range':['Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18'],
    'approved_sem1_range':['Rango >10','Rango 6-10','Rango 3-5','Rango >10','Rango 6-10','Rango 0-2'],
    'grade_admission_range':['Rango 130-145','Rango 100-115','Rango 145-160','Rango 130-145','Rango 145-160','Rango 100-115'],
    'imagen': [
        "https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg?w=740&t=st=1695773151~exp=1695773751~hmac=c5e7f88a86b317d607c4521d19c82847adcd45cb57a4214c7dd9f9f7c5a48e86",
        "https://img.freepik.com/free-photo/medium-shot-smiley-guy-with-crossed-arms_23-2148227980.jpg?w=740&t=st=1695773183~exp=1695773783~hmac=7ad00f735040d1e74ed3089f70072fefc083c2143e99802e621aaf430aa7d1f1",
        "https://img.freepik.com/free-photo/happy-man_1368-1596.jpg?w=360&t=st=1695773202~exp=1695773802~hmac=8d3a6a8ed84d6e3918c304ccf690356643eb2a4c7042e1be0bf2e72331b92dca",
        "https://img.freepik.com/free-photo/young-handsome-guy-carrying-boxes_144627-25941.jpg?w=360&t=st=1695773260~exp=1695773860~hmac=df2a0ace2b2e03b6316b07f72fd344c2e9dbba229e1ed08f0e5f6b2906c0cfa6",
        "https://img.freepik.com/free-photo/portrait-beautiful-woman-posing_23-2148723157.jpg?w=740&t=st=1695773283~exp=1695773883~hmac=93b9536b0392f56efc9804792a5a03de5de80e4d572c833eff5498630e4794c1",
        "https://img.freepik.com/free-photo/beautiful-smiling-woman-with-long-hair-standing-against-blue-background_662251-521.jpg?w=740&t=st=1695773296~exp=1695773896~hmac=8836535e3a5128da8634054cfbb2236505b72d6c6b42a5beb43319092394de52"
    ],
}

df=df = pd.DataFrame(data)


# Creamos el dashboard
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.layout = html.Div([
    html.H1('Estimación de Riesgo de Deserción - Estudiantes Antiguos',
            style={
            'textAlign': 'center',
            'color': colors['text']
        }),
    dcc.Input(id='codigo-input', type='number', placeholder='Código de estudiante'),
    html.Button('Buscar', id='buscar-button'),
    html.Div(id='nombre-output'),
    html.Img(id='imagen-output', src='',style={'width': '200px', 'height': '200px'}),
    html.Div(id='probabilidad-output'),
])
@app.callback(
    [Output('nombre-output', 'children'), Output('imagen-output', 'src'),Output('probabilidad-output', 'children')],
    [Input('buscar-button', 'n_clicks')],
    [dash.dependencies.State('codigo-input', 'value')]
)
def buscar_estudiante(n_clicks, codigo):
    if not codigo:
        return '', 'https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png?20091205084734',''

    estudiante = df[df['Codigo'] == codigo]

    if estudiante.empty:
        return 'Estudiante no encontrado', 'https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png?20091205084734',''

    nombre = estudiante.iloc[0]['nombre']
    imagen = estudiante.iloc[0]['imagen']
    col = data['Codigo'].index(codigo)
    estudiante =  {
        'Gender':data['Gender'][col],
        'Displaced':data['Displaced'][col],
        'Scholarship holder':data['Scholarship holder'][col],
        'Debtor':data['Debtor'][col],
        'Tuition fees up to date':data['Tuition fees up to date'][col],
        'age_range':data['age_range'][col],
        'approved_sem1_range':data['approved_sem1_range'][col],
        'grade_admission_range':data['grade_admission_range'][col],
    }
    estudiante
    caso = infer.query(["Target"], evidence=estudiante)
    probabilidad_retiro = caso.values[0] 
    return f'Nombre del estudiante: {nombre}', imagen, f'Probabilidad de retirarse: {probabilidad_retiro:.2f}%'

if __name__ == '__main__':
    app.run_server(debug=True)
