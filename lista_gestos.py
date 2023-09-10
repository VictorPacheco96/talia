import os

lista_nomes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "Meu",
    "Meu nome é",
    "O",
    "Ola",
    "P",
    "Pedro",
    "Projeto",
    "R",
    "Revolucionario",
    "Seus"
]

gestos = os.listdir(os.path.join('Gestos'))

for gesto in gestos:
    print(f'"{gesto}",')