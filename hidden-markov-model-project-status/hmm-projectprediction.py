import pandas as pd
import numpy as np
from hmmlearn import hmm
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


def baum_welch(edataInput, n_states=4):
    model = hmm.CategoricalHMM(n_components=n_states)

    # Estimate the model parameters
    model.fit(edataInput)

    return model


def encode_states(csv_file):
    # Read the data from the CSV file
    data = pd.read_csv(csv_file, header=None)
    data = data.values.tolist()
    encoded_data = []

    for row in data:
        # Each row in `data` is a list of three elements: [c, t, q]
        encoded_state = row[0] * 9 + row[1] * 3 + row[2]
        encoded_data.append(encoded_state)

    # Convert the list to a numpy array and reshape it for the hmmlearn library
    encoded_data = np.array(encoded_data).reshape(-1, 1)

    return encoded_data


def predict_observations(model, hidden_states):
    B = model.emissionprob_
    predicted_observations = np.argmax(B[hidden_states], axis=1)

    return predicted_observations


def run_test(dataset, test_data, true_data, testitr):
    # test prediction and find best suitable
    found_model = baum_welch(dataset)
    found_accuracy = 0
    for i in range(0, testitr):
        model = baum_welch(dataset)
        hidden_states = model.predict(test_data)
        predicted_observations = predict_observations(model, hidden_states)
        correct = 0
        total = len(true_data)
        for j in range(total):
            if predicted_observations[j] == true_data[j]:
                correct += 1
        accuracy = correct / total
        if accuracy >= 0.4 and accuracy > found_accuracy:
            found_model = model
            found_accuracy = accuracy

    return found_model, found_accuracy
