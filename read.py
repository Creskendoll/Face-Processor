from misc import Reader
from os.path import abspath

file = abspath("./recording.csv")
reader = Reader(file)

for row in reader.read_csv():
    landmarks, face_index = row[:-1], eval(row[-1])
    landmarks = [eval(l) for l in landmarks]
    print("Face index: {}\nLandmarks: {}".format(face_index, landmarks))