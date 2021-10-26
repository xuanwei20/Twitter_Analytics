import pandas as pd
from nltk.tokenize import word_tokenize

class MLAnalyzeData:
    def __init__(self):
        self.classifier = None
        self.prepare_prediction()

    def read_training_data_file(self, file_name='train.csv'):
        """Read training data from file instead of db"""
        raw_dataset = pd.read_csv(file_name)
        twitter = raw_dataset.iloc[:, 3]
        decisions = raw_dataset.iloc[:, 4]
        return twitter, decisions
        # print(twitter)
        # print(decisions)

    def prepare_training_matrix(self, training_data_file_name='train.csv'):
        with open('disaster_words.txt') as file:
            raw_content = file.read()
            contents_1 = raw_content.lower().split()

        with open('relevant_words.txt') as file:
            raw_content = file.read()
            contents_2 = raw_content.lower().split()

        filename = 'matrix.csv'
        with open(filename, 'w') as file_object:
            file_object.write(f"tweet number, number of disaster words, number of relevant words, decision\n")

        twitter, decisions = self.read_training_data_file(training_data_file_name)
        n = 0
        for tweet, decision in zip(twitter, decisions):
            n = n + 1
            tokens = word_tokenize(tweet)
            # print(f"{n}. {tokens}")

            i = 0
            j = 0
            for token in tokens:
                token = token.lower()
                if token in contents_1:
                    i = i + 1
                    # print(f"The word \"{token}\" is a disaster word.")
                elif token in contents_2:
                    j = j + 1
                    # print(f"The word \"{token}\" is a relevant word.")
                else:
                    continue
                    # print(f"The word \"{token}\" is neither a disaster word nor a relevant word.")

            # print(f"number of disaster words: {i}")
            # print(f"number of relevant words: {j}\n")

            with open(filename, 'a') as file_output:
                file_output.write(f"{n}, {i}, {j}, {decision}\n")

    def load_training_data_matrix(self):
        # Import dataset
        dataset = pd.read_csv('matrix.csv')
        x = dataset.iloc[:, [1, 2]].values
        y = dataset.iloc[:, 3].values
        return x, y

    def logistic_regression(x, y):
        # Split the dataset into training dataset and testing dataset
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=None)

        # Remove mean and scale to unit variance
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Adjust logistic regression to the training dataset
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state=0)
        classifier.fit(x_train, y_train)

        return classifier, x_test, y_test

    def analyze_data(self, x_test):
        # Predict the testing dataset decisions
        if self.classifier is None:
            raise Exception("A classifier needs to be calculated before using the analyze_data function")
        y_predict = self.classifier.predict(x_test)
        return y_predict

        # Make confusion matrix
        # from sklearn.metrics import confusion_matrix
        # cm = confusion_matrix(y_test, y_predict)

    def print_accuracy(self, y_test, y_predict):
        # Evaluate model accuracy
        m = 0
        l = 0
        for t, p in zip(y_test, y_predict):
            if t == p:
                m = m + 1
            else:
                l = l + 1
        print(f"Matched decision number: {m}, Unmatched decision number: {l}, Total decision number: {m + l}")
        print(f"Accuracy: {m / (m + l)}")

    def prepare_prediction(self):
        self.prepare_training_matrix()
        x, y = self.load_training_data_matrix()
        self.classifier, x_test, y_test = self.logistic_regression(x, y)
        y_predict = self.analyze_data(x_test)
        self.print_accuracy(y_test, y_predict)


if __name__ == '__main__':
    # Test complete run for test data
    #read_training_data_file(file_name='train.csv')
    mla = MLAnalyzeData()
