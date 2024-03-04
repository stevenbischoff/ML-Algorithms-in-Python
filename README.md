# ML-Algorithms-in-Python
Machine Learning classification and regression algorithms coded for the Spring 2023 Machine Learning course at Johns Hopkins. Algorithms are coded "from scratch", i.e. using only Pandas, NumPy, and built-in libraries.

## Neural Networks

This folder contains classes for single-layer, multi-layer, and autoencoder neural networks for both classification and regression tasks. The models.py module contains abstract classifier and regressor parent classes.

## Non-Parametric

This folder contains classes for K-Nearest Neighbors for both classification and regression tasks. The classes can handle both numeric and categorical input data. Distance between numeric data points is measured using Euclidean distance, while distance between categorical data points is measured using the Value Difference Metric (Stanfill and Waltz 1986, Toward Memory-Based Reasoning).

The classes also include methods that implement Edited and Condensed KNN, which help alleviate the computational expense of KNN on large datasets.

## Reinforcement Learning

algorithms.py contains a learner class that uses reinforcement learning to quickly steer a simulated car around a track without crashing. The class implements three different learning algorithms: value iteration, Q-Learning, and SARSA. 

## Decision Trees

TBA
