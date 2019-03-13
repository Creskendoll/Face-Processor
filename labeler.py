import pandas as pd
from Face import Face

data = pd.read_csv("data.csv")

columns = data.columns
faces = []

for d in data.values:
    faces.append(Face(d, columns))

for f in faces:
    print(f.getFacePosition())