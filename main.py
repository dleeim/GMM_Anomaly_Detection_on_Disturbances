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
    y = data['label'].to_numpy()
    X_df = data.drop("label", axis=1) 
    # Preprocess data
    X_scaled = scaler.transform(X_df)          
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


def component_variances(gmm):
    """
    Return per-component per-feature variances: shape (K, D).
    """
    cov_type = gmm.covariance_type
    covs = gmm.covariances_
    K, D = gmm.n_components, gmm.means_.shape[1]

    if cov_type == "full":
        return np.stack([np.diag(c) for c in covs], axis=0)
    if cov_type == "tied":
        return np.tile(np.diag(covs), (K, 1))
    if cov_type == "diag":
        return covs
    if cov_type == "spherical":
        return np.tile(np.asarray(covs).reshape(-1, 1), (1, D))
    raise ValueError(f"Unknown covariance_type: {cov_type}")


def explain_predicted_anomaly(
    df_train, df_test,
    X_train_scaled, X_test_scaled,
    y_test, y_pred,
    model,
    sample_idx=None
):
    gmm, tau = model
    feature_names = [c for c in df_train.columns if c != "label"]

    ll_train = gmm.score_samples(X_train_scaled)
    ll_test  = gmm.score_samples(X_test_scaled)

    # pick a sample that was predicted anomaly
    if sample_idx is None:
        candidates = np.where(y_pred == 1)[0]
        if len(candidates) == 0:
            # fallback: pick the most abnormal (lowest likelihood) point anyway
            sample_idx = int(np.argmin(ll_test))
        else:
            # pick the strongest anomaly for a clear story
            sample_idx = int(candidates[np.argmin(ll_test[candidates])])

    x = X_test_scaled[sample_idx]
    ll_x = ll_test[sample_idx]

    # component responsibility (regime)
    resp = gmm.predict_proba(x.reshape(1, -1))[0]
    k_star = int(np.argmax(resp))

    # per-feature z-scores relative to the selected regime/component
    mu = gmm.means_[k_star]
    var = component_variances(gmm)[k_star]
    z = (x - mu) / np.sqrt(var + 1e-12)

    order = np.argsort(np.abs(z))[::-1]
    top_idx = order[0]
    top_feats = feature_names[top_idx]
    top_z = z[top_idx]

    # ---- PLOTTING ----
    fig, ax = plt.subplots(1, 1)

    # one-feature distribution for the most deviated feature
    f0 = top_feats
    x0 = df_test.iloc[sample_idx][f0]
    ax.hist(df_train[f0].to_numpy(), bins=60, density=True, alpha=0.55, label="Train (normal)")
    ax.hist(df_test.loc[df_test["label"] == 1, f0].to_numpy(), bins=60, density=True, alpha=0.35, label="Test anomalies (label=1)")
    ax.axvline(x0, linewidth=2, label=f"Selected value = {x0:.3g}")
    ax.set_title(f"(C) Most deviated feature: '{f0}' (original scale)")
    ax.set_xlabel(f0)
    ax.legend()

    plt.tight_layout()
    plt.show()

    print(f"sample_idx={sample_idx}, label={int(y_test[sample_idx])}, pred={int(y_pred[sample_idx])}, LL={ll_x:.6f}, tau={tau:.6f}")

    return sample_idx, top_feats


def main():

    #Load the data
    scaler = fit_preprocess('Data/data_train.csv')
    X_train_scaled, y_train = load_and_preprocess('Data/data_train.csv', scaler)
    X_test_scaled, y_test = load_and_preprocess('Data/data_test.csv', scaler)

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

    df_train = pd.read_csv("Data/data_train.csv")
    df_test  = pd.read_csv("Data/data_test.csv")
    idx, feat = explain_predicted_anomaly(df_train, df_test, X_train_scaled, X_test_scaled, y_test, y_pred, model)


if __name__ == '__main__':
    main()