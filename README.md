
# Gender Classification using Deep Learning

This project involves the creation and training of a deep learning model to classify images by gender. The dataset used for training includes a variety of pre-processed images which have been augmented to improve the model's ability to generalize.

## Setup Instructions

To run this project, you need Python installed on your system. The project has been tested on Python 3.8. Additionally, you need to install several dependencies which include TensorFlow, Pandas, Matplotlib, NumPy, and scikit-learn.

### Dependencies

- TensorFlow 2.6.0
- Pandas 1.3.3
- Matplotlib 3.4.3
- NumPy 1.21.2
- scikit-learn 0.24.2

You can install all the necessary dependencies by running the following command:

```
pip install -r requirements.txt
```

### Dataset

The dataset is split into training and testing data. The training data is located within a zipped file `train_nLPp5K8.zip`, which needs to be unzipped before use.

### Training the Model

To train the model, you will run one of the Jupyter notebooks or Python scripts provided. The scripts are iterative versions, with each version potentially introducing improvements or changes in the model architecture or training process.

### Model Architecture

The model architecture includes convolutional layers followed by max-pooling layers, flattening, and dense layers with dropout for regularization.

### Training Process

Training involves the use of data generators for loading and augmenting images in real-time. Various configurations of data augmentation are tested in different iterations. Callbacks such as `EarlyStopping` and `ModelCheckpoint` are used to monitor training and save the best model.

### Evaluation and Predictions

After training, the model's performance is evaluated using accuracy and loss plots for both training and validation data. Predictions are made on the testing dataset, and the results are saved to a CSV file for submission.

## Running the Code

To start training the model, navigate to the project directory and start Jupyter Notebook or run the Python script using the following commands:

For Jupyter Notebook:

```
jupyter notebook <notebook_name>.ipynb
```

For Python script:

```
python <script_name>.py
```

Replace `<notebook_name>` and `<script_name>` with the actual names of the notebook or script you wish to run.

## Structure

The project directory is structured as follows:

- `images/` - Contains the dataset images after unzipping `train_nLPp5K8.zip`.
- `notebook/` - Contains Jupyter notebooks for iterative model development.
- `source/` - Contains Python scripts for the project.
- `requirements.txt` - Lists all the dependencies for the project.
- `README.md` - Provides an overview and instructions for the project.

For more details on the implementation, please refer to the individual notebooks and source code files.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
