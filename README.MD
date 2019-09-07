# Working with ML.NET
Welcome to your first experience with ML.NET!

ML.NET is an open-source cross-platform library to do machine learning in .NET.
This repository aims to showcase how ML.NET can be use to predict the survival rate of specific passengers on the Titanic

## Machine Learning in .NET
Machine learning (ML) is the scientific study of algorithms and statistical models that computer systems use to effectively perform a specific task without using explicit instructions, relying on patterns and inference instead. 

Machine Learning algoritms are mostly written in either of the two most popular languages, R and Python. ML.NET opens the door to quickly get up to speeed with Machine Learning for .NET developers, and incorporate the models in an existing application.

To learn more about ML.NET, make sure to visit their [GitHub repo](https://github.com/dotnet/machinelearning)

## Getting started
This repo consist of a simple .NET Core app used to train a binary classification model in order to predict survival outcome on the Titanic. In addition, this repo also contains a simple Azure Function which can be used to deploy the trained model.

Among other things, this application demonstrates the following features of ML.NET
- Using ML.NET to set up a training pipeline
- Transforming data and applying data pre-processing 
	- Replace missing values
	- Variable encodings
- Training a model using binary classification
- Evaluating the trained model and saving it as a .zip file
- Encapsulating the trained model in a scalable Azure Function 

### Workflow
![ML.NET workflow](https://github.com/excellaco/dotnet-machinelearning-titanic/blob/master/ml-workflow.png)

### Train the model
1. Clone the repo locally
2. Open the solution and set the TitanicPredictions.Train project as startup project
3. Run the console app
4. If desired, alter the learner or modify the data set to improve accuracy of the model

### Predict
1. Clone the repo locally
2. Open the solution and set the TitanicPredictions.Predict project as startup project
3. Run the Azure Function (or deploy to Azure service)
4. Using Postman, call the Azure Function with the following body:

```
{
	"Pclass" : "1",
	"Name" : "Mr. Smith, James",
	"Sex": "male",
	"Age": "32",
	"SibblingsAboard": "0",
	"ParentsAboard": "0"
}
```

## Contribute
See anything you want to improve? Do you want to build out the existing code base? Don't heistate to open a PR!
Please read our [contributing guidelines](https://github.com/excellaco/dotnet-machinelearning-titanic/blob/master/CONTRIBUTING.md)

