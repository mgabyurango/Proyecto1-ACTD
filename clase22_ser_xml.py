from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import XMLBIFWriter

# Defina la estructura de la red
modelo = BayesianNetwork([("C", "A"), ("U", "A")])

# Defina las CPDs:
cpd_c = TabularCPD(variable="C", variable_card=3, values=[[0.33], [0.33], [0.33]])
cpd_u = TabularCPD(variable="U", variable_card=3, values=[[0.33], [0.33], [0.33]])
cpd_a = TabularCPD(
    variable="A",
    variable_card=3,
    values=[
        [0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
        [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
        [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0],
    ],
    evidence=["C", "U"],
    evidence_card=[3, 3],
)

# Asocie las CPDs con la estructura de red
modelo.add_cpds(cpd_c, cpd_u, cpd_a)

# Some other methods
print(modelo)


# check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo.check_model()


# write model to an XML BIF file 
writer = XMLBIFWriter(modelo)
writer.write_xmlbif('monty.xml')




