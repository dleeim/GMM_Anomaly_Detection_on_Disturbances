import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
def fit_preprocess(data_path):
    """
    This function reads the data from a csv file and returns the standard scaler object.
    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.

    Returns
    -------
    scaler : StandardScaler Object
        A standard Scaler fit to the training data
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Remove label for scaler
    data = data.drop('label', axis=1)
    # Fit StandardScaler and return
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def load_and_preprocess(data_path, scaler):
    """
    Return standardized data and true labels from the training data csv file.

    Parameters
    ----------
    data_path : str
        Path to the csv file containing the data.
    scaler : StandardScaler Object
        A standard Scaler fit to the training data

    Returns
    -------
    X_scaled : 2D numpy array
        Preprocessed data.
    y : 1D numpy array
        True normal/anomaly labels of the data.
    """
    # Load from file
    data = pd.read_csv(data_path)
    # Separate label and choose features, then reshape to 2D array
    X = data.drop('label', axis=1).to_numpy()
    y = data['label'].to_numpy()
    # Preprocess data
    X_scaled = scaler.transform(X)
    return X_scaled, y

def fit_model(X):
    """
    Accepts a 2D numpy array and returns a fitted anomaly detection model. The model is a Gaussian Mixture Model (GMM).
    The number of components is determined by the Bayesian Information Criterion (BIC).

    Parameters
    ----------
    X : 2D numpy array.
        Already preprocessed data used to fit the model.

    Returns
    -------
    model : tuple
        Contains model object and threshold value.
    """
    n_components_range = range(1, 6)
    bic_scores = []
    model = None  # Initialize the variable to store the best model
    lowest_bic = float('inf')  # Initialize the lowest BIC as infinity

    # Loop over the range of components and fit GMM
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(X)  # Use scaled training data
        bic = gmm.bic(X)  # Calculate BIC score
        bic_scores.append(bic)

        # Update the best model if the current BIC is lower than the previous lowest BIC
        if bic < lowest_bic:
            lowest_bic = bic
            model = gmm
    
    #Get threshold
    # Calculate log likelihoods from training points
    log_likelihoods_train = model.score_samples(X)

    # Define the anomaly detection threshold based on training data
    threshold_percentile = 5  # Use the 5th percentile
    threshold = np.percentile(log_likelihoods_train, threshold_percentile)

    model = (model, threshold)
    return model
    

def predict(X, model):
    """
    Takes in the fitted model as well as the preprocessed data and returns the predictions.
    A threshold value is first calculated using the 5th percentile of the log likelihoods of the training data.

    Parameters
    ----------
    X : 2D numpy array.
        Preprocessed data.
    model : object
        Anomaly detection model.

    Returns
    -------
    y_pred : 1D numpy array.
        Array of normal/anomaly predictions.
    """

    # Predict on test data using the threshold
    log_likelihoods_test = model[0].score_samples(X)
    threshold = model[1]
    y_pred = (log_likelihoods_test < threshold).astype(int)  # Anomaly if log likelihood < threshold
    return y_pred

def main():

    #Load the data
    scaler = fit_preprocess('data_train.csv')
    X_train_scaled, y_train = load_and_preprocess('data_train.csv', scaler)
    X_test_scaled, y_test = load_and_preprocess('data_test.csv', scaler)

    # Fit the model
    model = fit_model(X_train_scaled)

    # Predict on the test data
    y_pred = predict(X_test_scaled, model)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

if __name__ == '__main__':
    main()