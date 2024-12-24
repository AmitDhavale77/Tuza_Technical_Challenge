# AI Engineering Internship: Technical Assessment

This project delivers all the required outputs to address the problem statement provided in the challenge. Additionally, it includes optional code to train the model and perform data analysis.

The Python version and library requirements are in requirements.txt

## Project Structure

#### `main.py`

This script organizes all the subsequent scripts into a full project pipeline.

#### `data_processing.py`

This script contains functions to preprocess the given data by performing multiple steps such as categorizing, cleaning, transforming,
encoding, and visualizing.

#### `processed_input_data.py`

This script contains the functions to preprocess the given input row/rows of data by performing multiple steps such as categorizing, cleaning, transforming,
encoding, and scaling.

#### `ml_model_supervised.py`

This script contains functions to execute the Random Forest classification pipeline including data preprocessing,
model training, evaluation, and optional saving of the model and label encoder.

#### `predict.py`

This script contains functions to predict the pricing category (e.g., 'Competitive', 'Neutral', 'Non-Competitive')
for the given input data based on a trained machine-learning model.

#### `data.csv`

This is the CSV file provided as part of the challenge.

#### `updated_transaction_data_with_labels.csv`

This CSV file contains the final processed attributes of the dataset.

#### `label_encoder.pkl`

This file contains stored label encodings for the target class ('Competitive', 'Neutral', 'Non-Competitive') class.
For example, encodings can be => {'Competitive': 0, 'Neutral': 1, 'Non-Competitive: 2'}

#### `random_forest_model.pkl`

This file contains the trained random forest classifier on the provided dataset "data.csv" in the challenge.

#### `scaler.pkl`

This file contains the stored scaling factors for the final processed numerical attributes in the dataset. This is required to scale the input row/rows before passing them to the trained model.

## Instructions

Running the script to output whether the fees a business currently pays are competitive, neutral or non-competitive.

```bash
python main.py
```

### Run Modes

#### run_data_preprocessing()

This will preprocess the input dataset, provide you with the data visualization and save it as a new CSV.

#### run_model_training()

This will train the Random Forest model using the preprocessed data obtained from run_data_preprocessing().
It also gives the K-fold cross-validation (K=5) evaluation of the trained model along with the visualization of the Confusion Matrix

#### make_predictions()

This will make predictions using the trained model.
Note: You can either pass a single row (pandas Series) or multiple rows (pandas Dataframe) as input.
