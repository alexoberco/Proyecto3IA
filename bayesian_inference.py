import json

# Función para cargar la red bayesiana desde archivo
def load_bayesian_network(file_path):
    with open(file_path, 'r') as f:
        network = json.load(f)
    return network

# Función para calcular la probabilidad conjunta
def calculate_joint_probability(network, event):
    joint_prob = 1.0
    for node, value in event.items():
        if node in network['dependencies']:
            parent_values = ','.join([event[parent] for parent in network['dependencies'][node]])
            key = f"{node}|{','.join(network['dependencies'][node])}"  # Ajustar la clave con la estructura correcta
            print(f"Intentando acceder a la clave: {key} -> {parent_values}")
            if parent_values in network['probabilities'][key]:
                prob = network['probabilities'][key][parent_values][value]
            else:
                print(f"Combinación {parent_values} no encontrada en las probabilidades.")
                return 0
        else:
            prob = network['probabilities'][node][value]
        joint_prob *= prob
    return joint_prob

# Algoritmo de inferencia bayesiana
def bayesian_inference(network, query, evidence):
    hidden_vars = [var for var in network['nodes'] if var not in query and var not in evidence]
    
    def recursive_inference(vars, evidence, query):
        if not vars:
            event = {**evidence, **query}
            return calculate_joint_probability(network, event)
        var = vars[0]
        total_prob = 0
        for value in network['nodes'][var]['values']:
            new_evidence = {**evidence, var: value}
            total_prob += recursive_inference(vars[1:], new_evidence, query)
        return total_prob

    query_prob = recursive_inference(hidden_vars, evidence, query)
    normalizing_constant = recursive_inference(hidden_vars, evidence, {})
    
    return query_prob / normalizing_constant

# Cargar la red bayesiana desde el archivo JSON
network = load_bayesian_network('bayesian_network.json')

# Ejemplo: calcular la probabilidad de llegar a la reunión (attend) dado que hay lluvia ligera y el tren está retrasado
query = {"Appointment": "attend"}
evidence = {"Rain": "light", "Train": "delayed"}

# Realizar la inferencia
prob = bayesian_inference(network, query, evidence)
print(f"P({query} | {evidence}) = {prob:.4f}")
