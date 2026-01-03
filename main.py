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
    """Return per-component per-feature variances: shape (K, D)."""
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

def normal_pdf(x, mu, var):
    var = np.maximum(var, 1e-12)
    return (1.0 / np.sqrt(2*np.pi*var)) * np.exp(-0.5 * ((x - mu)**2) / var)

def plot_gmm_1d_feature_overlay(
    df_train, df_test, feature_name,
    gmm, scaler,
    show_test="anomalies",   # "all" | "anomalies" | "pred_anomalies"
    y_pred=None,
    bins=60,
    grid_n=500
):
    # feature index
    feature_names = [c for c in df_train.columns if c != "label"]
    i = feature_names.index(feature_name)

    # data in original units
    x_train = df_train[feature_name].to_numpy()
    if show_test == "all":
        x_test = df_test[feature_name].to_numpy()
        test_label = "Test (all)"
    elif show_test == "anomalies":
        x_test = df_test.loc[df_test["label"] == 1, feature_name].to_numpy()
        test_label = "Test (label=1 anomalies)"
    elif show_test == "pred_anomalies":
        if y_pred is None:
            raise ValueError("Provide y_pred if show_test='pred_anomalies'")
        x_test = df_test.loc[np.asarray(y_pred)==1, feature_name].to_numpy()
        test_label = "Test (predicted anomalies)"
    else:
        raise ValueError("show_test must be 'all', 'anomalies', or 'pred_anomalies'")

    # build plotting grid (original units)
    lo = np.nanmin(np.concatenate([x_train, x_test])) if len(x_test) else np.nanmin(x_train)
    hi = np.nanmax(np.concatenate([x_train, x_test])) if len(x_test) else np.nanmax(x_train)
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, grid_n)

    # Convert GMM params (trained in scaled space) back to original units for this feature
    mu_scaled = gmm.means_[:, i]                      # (K,)
    var_scaled = component_variances(gmm)[:, i]       # (K,)
    scale = scaler.scale_[i]
    mean0 = scaler.mean_[i]

    mu_raw = mu_scaled * scale + mean0
    var_raw = var_scaled * (scale**2)

    # mixture + components in raw space
    weights = gmm.weights_
    comp_pdfs = np.array([weights[k] * normal_pdf(grid, mu_raw[k], var_raw[k]) for k in range(gmm.n_components)])
    mix_pdf = comp_pdfs.sum(axis=0)

    # plot
    plt.figure()
    plt.hist(x_train, bins=bins, density=True, alpha=0.45, label="Train (normal)")
    if len(x_test):
        plt.hist(x_test, bins=bins, density=True, alpha=0.35, label=test_label)

    # each component (weighted)
    for k in range(gmm.n_components):
        plt.plot(grid, comp_pdfs[k], linewidth=1, alpha=0.8, label=f"Gaussian {k+1}")

    # total mixture
    plt.plot(grid, mix_pdf, linewidth=3, label="GMM mixture (train-fit)")

    plt.title(f"1D marginal GMM overlay on feature: {feature_name}")
    plt.xlabel(f"{feature_name} (original units)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.show()


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

    print(f'No of component: {model[0].n_components}')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Select feature that you want to visualize how GMM detects disturbances
    FEATURE_NAME = "XMEAS(3)"
    df_train = pd.read_csv("Data/data_train.csv")
    df_test  = pd.read_csv("Data/data_test.csv")
    plot_gmm_1d_feature_overlay(df_train, df_test, FEATURE_NAME, model[0], scaler, show_test="anomalies", y_pred=y_pred)


if __name__ == '__main__':
    main()