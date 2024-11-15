from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob
import sys
import glob
import csv

DESCRIPTION = 5
CATEGORY = 2
AVOIDABLE = 3
ORDINARY = 4

def norm(string):
    return string.replace(',', '').replace("'", '')


class ExpenseClassifier:

    def __init__(self):
        training_data = self._load_data("data")
        self.category_classifier  = NaiveBayesClassifier([(x[0], x[1]) for x in  training_data])
        self.avoidability_classifier = NaiveBayesClassifier([(x[0], x[2]) for x in  training_data])
        self.ordinary_classifier =  NaiveBayesClassifier([(x[0], x[3]) for x in  training_data])

    def classify(self, description):
        res = {}
        res['category'] = self.category_classifier.classify(description)
        res['avoidable'] = self.avoidability_classifier.classify(description)
        res['ordinary'] = self.ordinary_classifier.classify(description)
        return res

    def accuracy(self):
        test_data = self._load_data("test")
        res = {}
        res['category'] = self.category_classifier.accuracy([(x[0], x[1]) for x in test_data])
        res['avoidable'] = self.avoidability_classifier.accuracy([(x[0], x[2]) for x in test_data])
        res['ordinary'] = self.ordinary_classifier.accuracy([(x[0], x[3]) for x in test_data])
        return res

    def _load_data(self, folder):
        data = []
        for f in glob.glob(folder + "/*.csv"):
            with open(f) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    if row[DESCRIPTION] and row[CATEGORY] and row[AVOIDABLE] and row[ORDINARY]:
                        data.append((norm(row[DESCRIPTION]), row[CATEGORY], row[AVOIDABLE], row[ORDINARY]))
        return data

class CSVEnricher:
    _data = []
    _enriched_data = []

    def __init__(self, file_name):

        with open(file_name) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                row[2] = norm(row[2])
                print(row[2])
                self._data.append(row)
        self.cl = ExpenseClassifier()

    def enrich(self):
        for row in self._data[2:]:
            if len(row) < 3:
               continue
            else:
                res = self.cl.classify(row[2])
                self._enriched_data.append(row[0:2] + [res['category'], res['avoidable'], res['ordinary']] + row[2:])

    def writeTo(self, file_name):
        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in self._enriched_data:
                writer.writerow(row)

def print_data(data):
    for x in data:
        print(x)

def main():
    enricher = CSVEnricher(sys.argv[1])
    enricher.enrich()
    enricher.writeTo("res.csv")

if __name__ == "__main__":
    main()
