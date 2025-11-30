"# Handwritten-Character-Classifier"

run pip install -r requirements.txt

To download and preprocess the data set run
python -m src.data

This will download and preprocess the dataset and store it in
data/processed

To train the cnn model run
python -m src.train

This trains the model over 15 epochs and store it in best_model.pt

To test the cnn model run
python -m src.test

This test best_model.pt and displays an accuaracy barchart.
Right now the best model strugles the most with "G", "I", and "L"


To train the Random Forest model run
python -m src.rf_train

To test the Random Forest model run
python -m src.rf_test

This tests best_rf_model.joblib and displays an accuracy barchart.

.gitignore prevents the dataset from being pushed to github

