import json

# Variables globales para almacenar la normalizing constant
normalizing_constant = 0

# Función para cargar la red bayesiana desde archivo
def load_bayesian_network(file_path):
    with open(file_path, 'r') as f:
        network = json.load(f)
    return network

# Función para calcular la probabilidad conjunta con depuración
def calculate_joint_probability(network, event):
    joint_prob = 1.0
    for node, value in event.items():
        if node in network['dependencies']:
            parent_values = ','.join([event[parent] for parent in network['dependencies'][node]])
            key = f"{node} | {','.join(network['dependencies'][node])}"
            # print(f"If: Intentando acceder a la clave: {key} -> {parent_values}")
            if parent_values in network['probabilities'][key]:
                prob = network['probabilities'][key][parent_values][value]
                print(f"P({node} = {value} | {','.join(network['dependencies'][node])} = {parent_values}) = {prob}")
            else:
                print(f"P({node} = {value} | {','.join(network['dependencies'][node])} = {parent_values}) = NOT Found in Network")
                return 0
        else:
            prob = network['probabilities'][node][value]
            print(f"P({node} = {value}) = {prob}")
        joint_prob *= prob
    return joint_prob

# Algoritmo de inferencia bayesiana para una pregunta específica
def bayesian_inference(network, query, evidence):
    hidden_vars = [var for var in network['nodes'] if var not in query and var not in evidence]
    
    def recursive_inference(vars, evidence, query):
        if not vars:
            event = {**evidence, **query}
            print()
            return calculate_joint_probability(network, event)
        var = vars[0]
        total_prob = 0
        for value in network['nodes'][var]['values']:
            new_evidence = {**evidence, var: value}
            total_prob += recursive_inference(vars[1:], new_evidence, query)
        return total_prob

    # Si la variable global de la constante de normalización es 0, calcularla
    global normalizing_constant
    if normalizing_constant == 0:
        print("-------------------------------------------------------------------------------------")
        print("Calculando la constante de normalización:")
        normalizing_constant = recursive_inference(hidden_vars, evidence, {})
        print(f"Constante de normalización = {normalizing_constant}")
    
    # Calcular la probabilidad de la query
    print("-------------------------------------------------------------------------------------")
    print(f"Calculando la probabilidad (sin normalizar) de {query} dado {evidence}:")
    query_prob = recursive_inference(hidden_vars, evidence, query)    
    print(f"P({query} | {evidence}) = {query_prob:.4f}")
    
    return query_prob / normalizing_constant

# Algoritmo para obtener la distribución completa de una variable
def distribution_of_variable(network, variable, evidence):
    dist = {}
    for value in network['nodes'][variable]['values']:
        query = {variable: value}
        prob = bayesian_inference(network, query, evidence)
        dist[value] = prob
    return dist

# Función para manejar la entrada del usuario
def main():
    # Pedir archivo JSON por consola
    file_path = input("Ingrese el nombre del archivo JSON con la red bayesiana: ")
    network = load_bayesian_network(file_path)

    # Imprimir las variables de la red y sus posibles valores
    print("------------------------------------------")
    print("Variables de la red bayesiana:")
    for node, data in network['nodes'].items():
        print(f"{node}: {data['values']}")
    print("------------------------------------------")

    # Imprimir posibles queries recorriendo queries de la red que tienen este formato
    # {
    #     "Opcion": "1 (Probabilidad)",
    #     "Variable": "Radiografia anormal",
    #     "Valor esperado": "Si",
    #     "Evidencia": "Fumador=Si"
    #   }
    print("Ejemplos de queries:")
    for query in network['queries']:
        print()
        for key, value in query.items():
            print(f"{key}: {value}")
    print("------------------------------------------")

    # Pedir tipo de pregunta
    print("Seleccione el tipo de pregunta:")
    print("1. Probabilidad de un evento específico")
    print("2. Distribución de una variable")
    question_type = input("Ingrese 1 o 2: ")

    if question_type == "1":
        # Pedir evento específico y evidencia
        variable = input("Ingrese la variable para la consulta: ")
        value = input(f"Ingrese el valor esperado para {variable}: ")
        evidence_input = input("Ingrese la evidencia en formato 'variable1=valor1,variable2=valor2': ")
        print()
        if evidence_input == "":
            evidence = {}
        else:
            evidence = dict(item.split("=") for item in evidence_input.split(","))
        query = {variable: value}

        # Realizar la inferencia
        prob = bayesian_inference(network, query, evidence)
        print("=====================================================================================")
        print("Resultado:")
        print(f"P({query} | {evidence}) = {prob:.4f}")

    elif question_type == "2":
        # Pedir variable y evidencia
        variable = input("Ingrese la variable para la distribución: ")
        evidence_input = input("Ingrese la evidencia en formato 'variable1=valor1,variable2=valor2': ")
        print()
        if evidence_input == "":
            evidence = {}
        else:
            evidence = dict(item.split("=") for item in evidence_input.split(","))

        # Obtener la distribución
        dist = distribution_of_variable(network, variable, evidence)
        print("=====================================================================================")
        print(f"Distribución de {variable} dado {evidence}:")
        for value, prob in dist.items():
            print(f"P({variable} = {value} | {evidence}) = {prob:.4f}")

if __name__ == "__main__":
    main()
