"""Implementations for ratio estimation classifiers."""

import abc

import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.preprocessing import StandardScaler


class Classifier(abc.ABC):
    """An abstract base class for a ratio estimation classifier."""

    @abc.abstractmethod
    def __init__(self):
        """Initialize a classifier."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit a classifier.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.
        y: np.ndarray (n_samples, )
            Target values, must be binary.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_log_likelihood_ratio(self, X):
        """Predict a log-likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        raise NotImplementedError

    def predict_likelihood_ratio(self, X):
        """Predict a likelihood ratio.

        Parameters
        ----------
        X: np.ndarray (n_samples, n_features)
            Feature vectors of data.

        Returns
        -------
        np.ndarray

        """
        return np.exp(self.predict_log_likelihood_ratio(X))

    @property
    @abc.abstractmethod
    def attributes(self):
        """Return attributes dictionary."""
        raise NotImplementedError


class LogisticRegression(Classifier):
    """A logistic regression classifier for ratio estimation."""

    def __init__(self, config=None, class_min=0):
        """Initialize a logistic regression classifier."""
        self.config = self._resolve_config(config)
        self.class_min = self._resolve_class_min(class_min)
        self.model = LogReg(**self.config)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """Fit a logistic regression classifier."""
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict_log_likelihood_ratio(self, X):
        """Predict a log-likelihood ratio."""
        Xs = self.scaler.transform(X)
        class_probs = np.maximum(self.model.predict_proba(Xs)[:, 1], self.class_min)
        return np.log(class_probs / (1 - class_probs))

    @property
    def attributes(self):
        """Return an attributes dictionary."""
        return {
            'parameters': {
                'coef_': self.model.coef_.tolist(),
                'intercept_': self.model.intercept_.tolist(),
                'n_iter': self.model.n_iter_.tolist()
            }
        }

    def _default_config(self):
        """Return a default config."""
        return {
            'penalty': 'l1',
            'solver': 'liblinear'
        }

    def _resolve_config(self, config):
        """Resolve a config for logistic regression classifier."""
        if not isinstance(config, dict):
            config = self._default_config()
        return config

    def _resolve_class_min(self, class_min):
        """Resolve a class min parameter that prevents negative inf values."""
        if isinstance(class_min, int) or isinstance(class_min, float):
            return class_min
        raise TypeError('class_min has to be either non-negative int or float')
