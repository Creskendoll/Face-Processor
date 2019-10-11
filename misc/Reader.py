import csv

class Reader(object):
    def __init__(self, file_path):
        self.file_path = file_path
    
    def read_csv(self):
        assert ".csv" in self.file_path, "File should be of type csv"
        with open(self.file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                yield row
    