# DTI-reviewer

Code repository for the manuscript "Validation guidelines for drug-target prediction methods".

## User Instructions

DTI-reviewer is a tool for classifying texts based on the following 16 categories: Machine Learning, Deep Learning, Attention-based, Docking, Classification, Regression, Cross-validation, Independent testing, SOTA comparison, Experimental validation, Accuracy, RMSE, AUROC, Spearman/Pearson, F1-score, and Other metric.
The binary classification is done with 16 fine-tuned models based on BioMed-RoBERTa [1].

To classify your texts, follow these steps:

1. Provide a single .csv file with your texts to be classified as input. The first line of the file should be the title, "text". See `example_input.csv` for reference.
2. Specify the relevant classification categories in the `user_prediction_config.json` file. Also specify the name of the input file and the save path.
3. Navigate to the `src/` directory, then on the command line run `python make_predictions.py`.

The code uses an ensemble of 5 category-specific models for prediction. As output, you will find the predictions by each individual model, as well as classifications based on majority votes for all provided text entries. Each category will output a separate file.

To run `make_predictions.py`, you require the following packages:
```
pandas==2.2.1
transformers==4.38.1
torch
```

## Model performance

![data_performance_combined](https://github.com/AronSchulman/DTI-reviewer/assets/63584295/3366cd95-63dc-43ee-9974-6380f24e350d)

**A**: The composition of training data for each document classifier, balanced to contain equal proportions of positive and negative documents as shown in green and red, respectively. **B**: The document classifiers were trained with a stratified five-fold cross-validation. Thus, for each document category, we train and evaluate five binary classifiers. The blue and orange bars show the average accuracies and binary F1-scores derived from the five classifiers, respectively, and the black vertical lines indicate the standard errors of the mean (SEM). Note: the y-axis starts from 0.80 to better show the differences between the 16 classification tasks.

## Hyperparameters

![hyperparams](https://github.com/AronSchulman/DTI-reviewer/assets/63584295/84a8c6fc-0100-476a-b8ce-4e9c7d67059b)

The search space and optimal hyperparameters for each binary classifier after 20 iterations of random search. Each iteration was 10 epochs in duration. The hyperparameter search space was kept the same for all the categories.

## References

[1] Gururangan, Suchin, et al. "Don't stop pretraining: Adapt language models to domains and tasks." arXiv preprint arXiv:2004.10964 (2020).
