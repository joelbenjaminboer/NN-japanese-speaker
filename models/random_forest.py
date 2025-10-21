"""
random_forest_model.py
---------------------------------
Wrapper class around scikit-learn's RandomForestClassifier
for the Japanese Vowels dataset.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class RandomForestWrapper:
	def __init__(self, n_estimators=200, max_depth=None, random_state=42):
		"""
		Initialize a RandomForest classifier with desired hyperparameters.
		"""
		self.model = RandomForestClassifier(
			n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=random_state
		)
		self.is_trained = False

	@staticmethod
	def load_npz(train_file, test_file):
		train = np.load(train_file)
		test = np.load(test_file)

		# Handle either normal or augmented naming
		if "X_train" in train:
			X_train, y_train = train["X_train"], train["y_train"]
		elif "X_augmented" in train:
			X_train, y_train = train["X_augmented"], train["y_augmented"]
		else:
			raise KeyError("No valid X/y keys found in training file")

		X_test, y_test = test["X_test"], test["y_test"]
		return X_train, y_train, X_test, y_test

	@staticmethod
	def flatten(X):
		"""
		Flatten a (samples, time, features) array to (samples, time*features)
		for use in classical ML models.
		"""
		return X.reshape(X.shape[0], -1)

	def train(self, X_train, y_train):
		"""
		Train the RandomForest model.
		"""
		print("Training Random Forest...")
		self.model.fit(X_train, y_train)
		self.is_trained = True
		print("Training complete.")

	def predict(self, X):
		"""
		Predict labels for input samples.
		"""
		if not self.is_trained:
			raise RuntimeError("Model not trained yet. Call `.train()` first.")
		return self.model.predict(X)

	def evaluate(self, X_test, y_test):
		"""
		Evaluate accuracy and return classification report.
		"""
		if np.all(y_test == -1):
			print("Test labels unknown")
			return None

		y_pred = self.predict(X_test)
		acc = accuracy_score(y_test, y_pred)
		print(f"Accuracy: {acc:.3f}")
		print("\nDetailed report:")
		print(classification_report(y_test, y_pred))
		return acc
