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
            children=[html.H1('Estudiantes Nuevos',style={'color': '#63815F','font-family': 'Arial'} )],
        ),
        # Creamos un contenedor para los inputs y para que devuelva la imagen y alguna información del estudiante
        html.Div(
            id="left-column",
            className="four columns",
            children=[ 
                # 1. Creamos el input para el rango de edad
                html.H6(children = 'Ingrese la edad del estudiante'),
                dcc.Dropdown(
                id='rangoedad',
                options=[{'label': 'NA', 'value': 'NA'}] + [{'label': i, 'value': i} for i in data_sel['age_range'].unique()],
                value='Rango 16-18'
                ),
                # 2. Creamos el input para el género
                html.H6(children = 'Ingrese el género del estudiante. En caso de no conocer la información ingrese NA.'),
                dcc.Slider(
                    id='gender',
                    min = 0,
                    max = 2, 
                    marks={0: 'Mujer', 1: 'Hombre', 2: 'NA'},
                    step = 1,
                    value=2),
                # 3. Creamos el input para si es desplazado
                html.H6(children = 'Ingrese si el estudiante es de otra ciudad. En caso de no conocer la información ingrese NA.'),
                dcc.Slider(
                    id='displaced',
                    min = 0,
                    max = 2, 
                    marks={0: 'No', 1: 'Si', 2: 'NA'},
                    step = 1,
                    value=2),
                # 4. Creamos el input para si es becado
                html.H6(children = 'Ingrese si el estudiante es becado o no.En caso de no conocer la información ingrese NA.'),
                dcc.Slider(
                    id='scholarshipholder',
                    min = 0,
                    max = 2, 
                    marks={0: 'No', 1: 'Si', 2: 'NA'},
                    step = 1,
                    value=2),
                # 5. Creamos el input para si es deudor
                html.H6(children = 'Ingrese si el estudiante es deudor o no. En caso de no conocer la información ingrese NA.'),
                dcc.Slider(
                    id='debtor',
                    min = 0,
                    max = 2, 
                    marks={0: 'No', 1: 'Si', 2: 'NA'},
                    step = 1,
                    value=2),
                # 6. Creamos el input para si tiene deudas hasta el momento.
                html.H6(children = 'Ingrese si el estudiante está al día con la matrícula. En caso de no conocer la información ingrese NA.'),
                dcc.Slider(id='tuitionfees',
                    min = 0,
                    max = 2, 
                    marks={0: 'No', 1: 'Si', 2: 'NA'},
                    step = 1,
                    value=2),
                # 7. Creamos el input para créditos aprobados en primer semestre
                html.H6(children = 'Seleccione el rango de créditos aprovados por el estudiante hasta el momento. En caso de no conocer la información ingrese NA.'),
                dcc.Dropdown(
                id='approvedcredits',
                options=[{'label': 'NA', 'value': 'NA'}] + [{'label': i, 'value': i} for i in data_sel['approved_sem1_range'].unique()],
                value='Rango 6-10'
                ),
                # 8.  Creamos el input para la callificación de admisión
                html.H6(children = 'Seleccione el rango de la nota de admisión del estudiante. En caso de no conocer la información ingrese NA.'),
                dcc.Dropdown(
                id='gradeadmission',
                options=[{'label': 'NA', 'value': 'NA'}]+[{'label': i, 'value': i} for i in data_sel['grade_admission_range'].unique()],
                value='Rango 115-130'
                ),
                html.Button('Estimar',
                            id= 'estimator-button')
            ],
            style={'width': '40%', 'display': 'inline-block','float':'left'}
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                html.Div(id='probabilidadAspirante'),
                html.Div(id='accuracy'),
                dcc.Graph(id='grafica'),
            ],
            style={'width': '55%', 'display': 'block','float':'left'}
        ),
    ],
)

# Realiza la estimación de la probabilidad en cascada al dar click en el botón
# ¿Qué recibe? Toda la información anterior
# ¿Qué devuelve? Una probabilidad de desertar
#                El accuracy del modelo y
#                Una gráfica comparandolo con el estudiante 'más común'
@app.callback(
    [Output('probabilidadAspirante', 'children'),Output('accuracy','children'),Output('grafica','figure')],
    [Input('estimator-button', 'n_clicks')],
    [dash.dependencies.State('rangoedad', 'value'),dash.dependencies.State('gender', 'value'),dash.dependencies.State('displaced', 'value'),dash.dependencies.State('scholarshipholder', 'value'),dash.dependencies.State('debtor', 'value'),dash.dependencies.State('tuitionfees', 'value'),dash.dependencies.State('approvedcredits', 'value'),dash.dependencies.State('gradeadmission', 'value')]
)
def estimar_probabilidad(n_clicks,rangoedad,gender,displaced,scholarshipholder,debtor,tuitionfees,approvedcredits,gradeadmission):
    if n_clicks==0:
        return ''
    # Regresa la tarjeta del estudiante
    aspirante = {}
    if rangoedad != 'NA':
        aspirante['age_range'] = rangoedad
    if gender != 2:
        aspirante['Gender'] = gender
    if displaced != 2:
        aspirante['Displaced'] = displaced
    if scholarshipholder != 2:
        aspirante['Scholarship holder'] = scholarshipholder
    if debtor != 2:
        aspirante['Debtor'] = debtor
    if tuitionfees != 2:
        aspirante['Tuition fees up to date'] = tuitionfees
    if approvedcredits != 'NA':
        aspirante['approved_sem1_range'] = approvedcredits
    if gradeadmission != 'NA':
        aspirante['grade_admission_range'] = gradeadmission
    # Hace la estimación 
    casoAspirante = infer.query(["Target"],evidence = aspirante)
    probabilidadAspirante = casoAspirante.values[0]*100

    # Calcula el accuracy del modelo
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
    accuracyScore = accuracy_score(y_true=testVariableTarget, y_pred=prediccion['Target'])*100

    # Crea la gráfica que lo compara con el estudiante más común
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
    data = [go.Bar(x=['Aspirante','Estudiante Común'], y=[probabilidadAspirante, probEstudianteComun], marker=dict(color=['lightskyblue','pink']))]
    layout = go.Layout(title=f'Probabilidad de retirar: Aspirante vs. Estudiante Común')
    return [f'La probabilidad de desertar del estudiante es de: {probabilidadAspirante:.2f}%'], f'Este modelo tiene un accuracy del: {accuracyScore:.2f}%',{'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
