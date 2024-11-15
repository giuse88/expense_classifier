import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from dateutil.parser import parse
import re
import joblib


def is_date(string):
    try:
        parse(string, fuzzy=False)  # Attempts to parse the string as a date
        return True
    except OverflowError:
        return False
    except ValueError:
        return False


def is_number(s):
    try:
        float(s)
        return True
    except OverflowError:
        return True
    except ValueError:
        return False


def remove_punctuation(text):
    # Define characters to remove
    pattern = r"[,\&\;\/\*\+\n\r\:\.\[\]\\]"
    # Use regex to replace defined characters with an empty string
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def cleaner(txt):
    tokens = remove_punctuation(txt).split(' ')
    tokens = [t.strip() for t in tokens if (not is_date(t) and not is_number(t))]
    tokens = [t for t in tokens if t != '']
    return ' '.join(tokens)


def select_training_columns(data):
    return data.drop(['Category', 'Date', 'Description', 'label_category'], axis=1)


def train():
    data = '/Users/giuseppepes/workspace/expense_classifier/data/training_data.csv'
    data = pd.read_csv(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Description'] = data['Description'].apply(lambda x: x.replace(',', ''))
    data['Description'] = data['Description'].apply(cleaner)
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(data['Description'])
    vectorized_df = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    le = LabelEncoder()
    data['label_category'] = le.fit_transform(data['Category'])

    X = data
    X = pd.concat([X, vectorized_df.reset_index(drop=True)], axis=1)
    y = data['label_category']

    # RANDOM_STATE = 42
    # TEST_SIZE = 0.2
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    # clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    # X_train_s, X_test_s = select_training_columns(X_train), select_training_columns(X_test)
    # clf.fit(X_train_s, y_train)
    # y_pred = clf.predict(X_test_s)
    # y_proba = clf.predict_proba(X_test_s)
    # y_pred_decoded = le.inverse_transform(y_pred)
    # # Evaluate the model
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'Accuracy of Decision Tree Classifier: {accuracy * 100:.2f}%')
    # print("Predicted categories:", y_pred_decoded)
    # X_test['predictions'] = y_pred_decoded
    # X_test['probability'] = y_proba.max(axis=1)
    # res = X_test[['Description', 'Amount', 'predictions', 'probability']]
    # res = pd.concat([res, data['Category'].loc[res.index]], axis=1).rename({'Category': 'Actual'}, axis=1)
    # res.to_csv("~/fit_res.csv")

    RANDOM_STATE = 42
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
    X_train_s = select_training_columns(X)
    clf.fit(X_train_s, y)
    joblib.dump(clf, '/Users/giuseppepes/workspace/expense_classifier/models/expense_classifier_202411.pkl')
    joblib.dump(le, '/Users/giuseppepes/workspace/expense_classifier/models/label_encoder.pkl')
    print('Train completed')


def test_model():
    ml, le = load_models()
    data = '/Users/giuseppepes/workspace/expense_classifier/data/training_data.csv'
    data = pd.read_csv(data)
    data['Date'] = pd.to_datetime(data['Date'])
    data['Description'] = data['Description'].apply(lambda x: x.replace(',', ''))
    data['Description'] = data['Description'].apply(cleaner)
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(data['Description'])
    vectorized_df = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    data['label_category'] = le.fit_transform(data['Category'])

    X = data
    X = pd.concat([X, vectorized_df.reset_index(drop=True)], axis=1)
    y = data['label_category']

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train_s, X_test_s = select_training_columns(X_train), select_training_columns(X_test)
    y_pred = ml.predict(X_test_s)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of Decision Tree Classifier: {accuracy * 100:.2f}%')


def categorize(data, model, le):
    # assert
    data['Date'] = pd.to_datetime(data['Date'])
    data['Description'] = data['Description'].apply(lambda x: x.replace(',', ''))
    data['Description'] = data['Description'].apply(cleaner)
    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(data['Description'])
    vectorized_df = pd.DataFrame(sparse_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    X = pd.concat([data.reset_index(drop=True), vectorized_df.reset_index(drop=True)], axis=1)
    X = X.reindex(model.feature_names_in_, fill_value=0, axis=1)
    y_pred = model.predict(X)
    y_pred_decoded = le.inverse_transform(y_pred)
    return y_pred_decoded


def load_models():
    return (joblib.load('/Users/giuseppepes/workspace/expense_classifier/models/expense_classifier_202411.pkl') ,
            joblib.load('/Users/giuseppepes/workspace/expense_classifier/models/label_encoder.pkl'))
