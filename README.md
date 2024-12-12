# Spam Classifier using ML.NET

## Overview

This project uses ML.NET to train a spam classifier model that can predict whether a given message is spam or not. The model is trained on a dataset of labeled messages and can be used to make predictions on new, unseen data.

## Dataset

The dataset used in this project is a CSV file containing labeled messages, where each message is either spam or not spam. The dataset is loaded into the ML.NET pipeline using the LoadFromTextFile method.

## Pipeline

### The pipeline consists of the following steps:

1. Text Featurization: The text data is featurized using the FeaturizeText method, which converts the text into a numerical representation that can be used by the machine learning algorithm.

2. Label Encoding: The label data is encoded using the MapValueToKey method, which converts the label into a numerical representation that can be used by the machine learning algorithm.

3. Multiclass Classification: The featurized data is then passed through a multiclass classification algorithm, which predicts the label of the message.

4. Prediction: The predicted label is then converted back into a string using the MapKeyToValue method.
Model Evaluation

The model is evaluated using the Evaluate method, which calculates the log loss and accuracy of the model on the test set.

## Prediction Engine

A prediction engine is created using the CreatePredictionEngine method, which can be used to make predictions on new, unseen data.

## Example Use Case

An example use case is provided in the Main method, where a sample message is passed through the prediction engine to determine whether it is spam or not.

## Requirements

.NET Core 3.1 or later
ML.NET 1.4 or later
## Getting Started

1. Clone the repository using git clone https://github.com/your-username/spam-classifier.git
2. Open the solution in Visual Studio

## Using the Program

### 1. To use the program, you will need to compile it first. 
Open a terminal and navigate to the directory where the project is located. Then, run the following command to compile the program:


    dotnet build

### 2. Running the Program
Once the program is compiled, you can run it using the following command:


    dotnet run
### 3. Passing a Sample Message
To test the program, you can pass a sample message as a command-line argument. For example:

    dotnet run "You won a prize! Click here to receive."

The program will process the message and print whether it is spam or not.

### 4. Program Output
The program output will be something like:

    Message: You won a prize! Click here to receive.
    Is Spam? yes
This indicates that the message was classified as spam by the program.

### 5. Using the Program with an Input File
You can also use the program with an input file that contains a list of messages. To do this, you will need to create a text file with the messages, one per line. Then, you can pass the file name as a command-line argument:


    dotnet run example.csv

The program will process the messages in the file and print whether they are spam or not.

Note: Make sure to replace example.csv with the actual name of your input file.


## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.