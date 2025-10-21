from pathlib import Path

from models.random_forest import RandomForestWrapper

if __name__ == "__main__":
	DATA_DIR = Path("data/processed_data")
	aug_file = DATA_DIR / "augmented_data.npz"
	train_file = DATA_DIR / "train_data.npz"
	test_file = DATA_DIR / "test_data.npz"

	# only the training on augmented data
	print("test on augmented training data")
	X_train, y_train, X_test, y_test = RandomForestWrapper.load_npz(aug_file, test_file)
	X_train = RandomForestWrapper.flatten(X_train)
	X_test = RandomForestWrapper.flatten(X_test)
	model = RandomForestWrapper(n_estimators=300, max_depth=30)
	model.train(X_train, y_train)
	y_pred = model.model.predict(X_test)
	print("Test Accuracy:", (y_pred == y_test).mean())
	model.evaluate(X_test, y_test)

	# training on original data only
	print("test on original training data only")
	X_train, y_train, X_test, y_test = RandomForestWrapper.load_npz(train_file, test_file)
	X_train = RandomForestWrapper.flatten(X_train)
	X_test = RandomForestWrapper.flatten(X_test)
	model = RandomForestWrapper(n_estimators=300, max_depth=30)
	model.train(X_train, y_train)
	y_pred = model.model.predict(X_test)
	print("Test Accuracy:", (y_pred == y_test).mean())
	model.evaluate(X_test, y_test)
