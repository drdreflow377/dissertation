import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve




def estimate_propensity_scores(X, y):
    logreg = LogisticRegression(fit_intercept=False)
    logreg.fit(X, y)
    propensity_scores = logreg.predict_proba(X)[:, 1]
    propensity_scores = np.maximum(propensity_scores, 1 - propensity_scores)  # Ensure non-negative
    return propensity_scores

def compute_class_weights(y, propensity_scores):
    mask_positives=np.array([i for i in y == 1])
    class_weights = {1: 1 / np.mean(propensity_scores[mask_positives])}  # Calculate class weight for positive class
    return class_weights

def train_pu_model(X, y, class_weights):
    pu_model = LogisticRegression(class_weight=class_weights)
    pu_model.fit(X, y)
    return pu_model



def calculate_optimal_threshold(y, y_probs):
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y, y_probs)

    # Calculate the distance to the top left corner of the ROC curve
    # calculates the square of the Euclidean distance from each point on the ROC curve to this ideal point.
    distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)

    # Get the threshold for the point with the minimum distance to the ideal point
    optimal_threshold = thresholds[np.argmin(distances)]

    return optimal_threshold
