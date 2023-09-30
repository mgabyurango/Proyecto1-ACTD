# Estimación de la Red
import plotly.express as px
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from ucimlrepo import fetch_ucirepo
from pgmpy.readwrite import BIFWriter
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
# Importar datos
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

data=pd.DataFrame.from_dict(predict_students_dropout_and_academic_success['data']['original'])
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

# Decidimos las variables a incluir en el modelo
variables_seleccionadas = ['Gender', 'Displaced', 'Scholarship holder', 'Debtor',
                           'Tuition fees up to date','age_range','approved_sem1_range','grade_admission_range','Target']
# Usamos únicamente las variables seleccionadas de la base de datos
data_sel = data_mod[variables_seleccionadas]

# Creamos el modelo
model = BayesianNetwork([("Gender", "Target"),("Displaced","Target"),
                         ("Scholarship holder","Target"),("Debtor","Target"),("Tuition fees up to date","Target"),
                         ("age_range","Target"),("approved_sem1_range","Target"),("grade_admission_range","approved_sem1_range")])
#Dividir el dataset entre prueba y entrenamiento
train, test = train_test_split(data_sel, test_size=0.2, random_state=98)
emv = MaximumLikelihoodEstimator(model=model, data=train)
model.fit(data=train, estimator = MaximumLikelihoodEstimator)
infer = VariableElimination(model)


#Evaluacion del modelo para predecir la variable Target
testVariableTarget = test['Target']
testVariableTarget.head()
testVariableTarget.value_counts()

testVariables = test.drop(columns=['Target'])
#testVariables = test.drop(columns=['Target','approved_sem1_range']) #En caso del modelo con arco adicional
testVariables.head()

prediccion=model.predict(testVariables) #Prediccion del modelo
prediccion.head() #Imprimir prediccion
prediccion.value_counts()
prediccion['Target'].value_counts()

# Importamos los datos de la Universidad como un dataframe
data_sel = pd.read_csv('dataSeleccionada.csv')
dfUniversidad = pd.read_csv('baseUniversidad.csv')

# Los guardamos como un diccionario
dictUniversidad = {
    'nombre': ['Juan Pablo Ríos Hernández', 'Samuel Felipe Ríos Parra', 'Felipe Sanabria Trimiño', 'Andrés Felipe Sanabria Cotrino', 'Mayerli Andrea Velandia León','María Gabriela Urango Llorente','Estudiante Promedio'],
    'Codigo':[201821819, 201821820,201821926,201815259,201817500,202013781,0],
    'Gender':[1,1,1,1,0,0,data_sel['Gender'].value_counts().idxmax()],
    'Displaced':[0,0,1,0,0,1,data_sel['Displaced'].value_counts().idxmax()],
    'Scholarship holder':[0,0,0,0,0,1,data_sel['Scholarship holder'].value_counts().idxmax()],
    'Debtor':[0,0,0,0,0,0,data_sel['Debtor'].value_counts().idxmax()],
    'Tuition fees up to date':[1,1,1,1,1,1,data_sel['Tuition fees up to date'].value_counts().idxmax()],
    'age_range':['Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18','Rango 16-18',data_sel['age_range'].value_counts().idxmax()],
    'approved_sem1_range':['Rango >10','Rango 6-10','Rango 3-5','Rango >10','Rango 6-10','Rango 0-2',data_sel['approved_sem1_range'].value_counts().idxmax()],
    'grade_admission_range':['Rango 130-145','Rango 100-115','Rango 145-160','Rango 130-145','Rango 145-160','Rango 100-115',data_sel['grade_admission_range'].value_counts().idxmax()],
    'imagen': [
        "https://img.freepik.com/free-photo/young-bearded-man-with-striped-shirt_273609-5677.jpg?w=740&t=st=1695773151~exp=1695773751~hmac=c5e7f88a86b317d607c4521d19c82847adcd45cb57a4214c7dd9f9f7c5a48e86",
        "https://img.freepik.com/free-photo/medium-shot-smiley-guy-with-crossed-arms_23-2148227980.jpg?w=740&t=st=1695773183~exp=1695773783~hmac=7ad00f735040d1e74ed3089f70072fefc083c2143e99802e621aaf430aa7d1f1",
        "https://img.freepik.com/free-photo/happy-man_1368-1596.jpg?w=360&t=st=1695773202~exp=1695773802~hmac=8d3a6a8ed84d6e3918c304ccf690356643eb2a4c7042e1be0bf2e72331b92dca",
        "https://img.freepik.com/free-photo/young-handsome-guy-carrying-boxes_144627-25941.jpg?w=360&t=st=1695773260~exp=1695773860~hmac=df2a0ace2b2e03b6316b07f72fd344c2e9dbba229e1ed08f0e5f6b2906c0cfa6",
        "https://img.freepik.com/free-photo/portrait-beautiful-woman-posing_23-2148723157.jpg?w=740&t=st=1695773283~exp=1695773883~hmac=93b9536b0392f56efc9804792a5a03de5de80e4d572c833eff5498630e4794c1",
        "https://img.freepik.com/free-photo/beautiful-smiling-woman-with-long-hair-standing-against-blue-background_662251-521.jpg?w=740&t=st=1695773296~exp=1695773896~hmac=8836535e3a5128da8634054cfbb2236505b72d6c6b42a5beb43319092394de52",
        ""
    ],
}


# Creamos la aplicación
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
# Agregamos un título para la aplicación
app.title = "Probabilidad de Deserción - Universidad"
# Definimos un servidor
server = app.server
app.config.suppress_callback_exceptions = True

# Diseñamos la aplicación
app.layout = html.Div(
    # Creamos un contenedor
    id="app-container",
    children=[
        # Agregamos un contenedor para el título de la página
        html.Div(
            id="banner",
            className="banner",
            children=[html.H1('Estudiantes Antiguos',style={'color': '#63815F','font-family': 'Arial'} )],
        ),
        # Creamos un contenedor para los inputs y para que devuelva la imagen y alguna información del estudiante
        html.Div(
            id="left-column",
            className="four columns",
            children=[ 
                dcc.Input(id='codigo-input', type='number', placeholder='Código de estudiante'),
                html.Button('Buscar', id='buscar-button'),
                html.Div(id='nombre-output'),
                html.Img(id='imagen-output', src='https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png?20091205084734',style={'width': '200px', 'height': '200px'}),
                html.H5(id='probabilidad-output'),
                html.Div(id='nuevaprob'),
                html.Div(id='probdisplaced'),
                html.H3(id='accuracyScore')
            ],
            style={'width': '40%', 'display': 'inline-block','float':'left'}
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                dcc.Graph(
                    id='grafica'
                )
            ],style={'width': '55%', 'display': 'block','float':'left'}
        ),
    ],
)


@app.callback(
    [Output('nombre-output', 'children'), Output('imagen-output', 'src'),Output('probabilidad-output', 'children'),Output('nuevaprob','children'),Output('probdisplaced','children'),Output('accuracyScore','children'),Output('grafica','figure')],
    [Input('buscar-button', 'n_clicks')],
    [dash.dependencies.State('codigo-input', 'value')]
)
def buscar_estudiante(n_clicks, codigo):
    if not codigo:
        return '', 'https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png?20091205084734','','','','',{'data': [], 'layout': {}}

    estudiante = dfUniversidad[dfUniversidad['Codigo'] == codigo]

    if estudiante.empty:
        return 'Estudiante no encontrado', 'https://upload.wikimedia.org/wikipedia/commons/5/59/Empty.png?20091205084734','','','','',{'data': [], 'layout': {}}

    nombre = estudiante.iloc[0]['nombre']
    imagen = estudiante.iloc[0]['imagen']
    col = dictUniversidad['Codigo'].index(codigo)
    estudiante =  {
        'Gender':dictUniversidad['Gender'][col],
        'Displaced':dictUniversidad['Displaced'][col],
        'Scholarship holder':dictUniversidad['Scholarship holder'][col],
        'Debtor':dictUniversidad['Debtor'][col],
        'Tuition fees up to date':dictUniversidad['Tuition fees up to date'][col],
        'age_range':dictUniversidad['age_range'][col],
        'approved_sem1_range':dictUniversidad['approved_sem1_range'][col],
        'grade_admission_range':dictUniversidad['grade_admission_range'][col],
    }
    caso = infer.query(["Target"], evidence=estudiante)
    probabilidad_retiro = caso.values[0]*100
    estudianteCopia =  {
        'Gender':dictUniversidad['Gender'][col],
        'Displaced':dictUniversidad['Displaced'][col],
        'Scholarship holder':dictUniversidad['Scholarship holder'][col],
        'Debtor':dictUniversidad['Debtor'][col],
        'Tuition fees up to date':dictUniversidad['Tuition fees up to date'][col],
        'age_range':dictUniversidad['age_range'][col],
        'approved_sem1_range':dictUniversidad['approved_sem1_range'][col],
        'grade_admission_range':dictUniversidad['grade_admission_range'][col],
    }
    # Estudiante si fuese debtor o no
    if estudianteCopia['Debtor'] == 1:
        estudianteCopia['Debtor'] = 0
    if estudianteCopia['Debtor'] == 0:
        estudianteCopia['Debtor']= 1
    # Hace una nueva inferencia
    casoNuevo = infer.query(["Target"],evidence = estudianteCopia)
    probabilidadNueva = casoNuevo.values[0]*100

    # Estudiante si cambio en scholarship holder
    estudianteCopia2 =  {
        'Gender':dictUniversidad['Gender'][col],
        'Displaced':dictUniversidad['Displaced'][col],
        'Scholarship holder':dictUniversidad['Scholarship holder'][col],
        'Debtor':dictUniversidad['Debtor'][col],
        'Tuition fees up to date':dictUniversidad['Tuition fees up to date'][col],
        'age_range':dictUniversidad['age_range'][col],
        'approved_sem1_range':dictUniversidad['approved_sem1_range'][col],
        'grade_admission_range':dictUniversidad['grade_admission_range'][col],
    }    
        # Estudiante si fuese otro becado
    if estudianteCopia2['Scholarship holder'] == 1:
        estudianteCopia2['Scholarship holder'] = 0
    if estudianteCopia2['Scholarship holder'] == 0:
        estudianteCopia2['Scholarship holder']= 1
    # Hace una nueva inferencia
    casoNuevo2 = infer.query(["Target"],evidence = estudianteCopia2)
    probabilidadNueva2 = casoNuevo2.values[0]*100
    accuracyScore = accuracy_score(y_true=testVariableTarget, y_pred=prediccion['Target'])*100

    # Calcula la probabilidad de dropout del estudiante más común
    estudianteComun = {
        'Gender':data_sel['Gender'].value_counts().idxmax(),
        'Displaced':data_sel['Displaced'].value_counts().idxmax(),
        'Scholarship holder':data_sel['Scholarship holder'].value_counts().idxmax(),
        'Debtor':data_sel['Debtor'].value_counts().idxmax(),
        'Tuition fees up to date':data_sel['Tuition fees up to date'].value_counts().idxmax(),
        'age_range':data_sel['age_range'].value_counts().idxmax(),
        'approved_sem1_range':data_sel['approved_sem1_range'].value_counts().idxmax(),
        'grade_admission_range':data_sel['grade_admission_range'].value_counts().idxmax(),        
    }

    casoEstudianteComun = infer.query(["Target"], evidence = estudianteComun)
    probEstudianteComun = casoEstudianteComun.values[0]*100
    
    # Crea la gráfica de barras del estudiante comun y el estudiante de interés
    data = [go.Bar(x=[nombre,'Estudiante Común'], y=[probabilidad_retiro, probEstudianteComun], marker=dict(color=['lightskyblue','pink']))]
    layout = go.Layout(title=f'Probabilidad de retirar: {nombre} vs. Estudiante Común')
    return nombre, imagen, f'Probabilidad de retirarse: {probabilidad_retiro:.2f}%', f'Al cambiar si el estudiante es deudor o no... Su probabilidad de retirarse sería: {probabilidadNueva:.2f}%', f'Al cambiar si el estudiante es becado o no... Su probabilidad de retirarse sería: {probabilidadNueva2:.2f}%',f'Este modelo tiene un accuracy del: {accuracyScore:.2f}%',{'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
