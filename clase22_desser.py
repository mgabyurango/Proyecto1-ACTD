from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.readwrite import BIFReader

# Read model from BIF file 
reader = BIFReader("monty.bif")
modelo = reader.get_model()

# Print model 
print(modelo)

# Check_model check for the model structure and the associated CPD and returns True if everything is correct otherwise throws an exception
modelo.check_model()

# Infering the posterior probability
from pgmpy.inference import VariableElimination

infer = VariableElimination(modelo)
posterior_p = infer.query(["C"], evidence={"U": 0, "A": 1})
print(posterior_p)

posterior_p2 = infer.query(["C"], evidence={"U": 0})
print(posterior_p2)

posterior_p3 = infer.query(["C"], evidence={"A": 1})
print(posterior_p3)

print(modelo.get_independencies())
