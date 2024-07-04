# DTI-reviewer

Code repository for the manuscript "Validation guidelines for drug-target prediction methods".

## User Instructions

DTI-reviewer is a tool for classifying texts based on the following 16 categories: Machine Learning, Deep Learning, Attention-based, Docking, Classification, Regression, Cross-validation, Independent testing, SOTA comparison, Experimental validation, Accuracy, RMSE, AUROC, Spearman/Pearson, F1-score, and Other metric.
The binary classification is done with 16 fine-tuned models based on BioMed-RoBERTa [1].

To use all scripts in the directory, you can construct a virtual environment:
1. On the command line, navigate to the `DTI-reviewer` directory.
2. Run `source setup.sh`

To deactivate the virtual environment, simply run `deactivate`.
To activate again, you can rerun `source setup.sh` (this will not restart the installation process).

To classify your texts, follow these steps:

1. Provide a single .csv file with your texts to be classified as input. See `3286_data_from_query.csv` for input reference. The only mandatory column is `text`.
2. Specify the path to the input data in the `config.yaml` file under PART IV in the file. Also specify the save path and the category from the above 16 options.
3. On the command line navigate to `DTI-reviewer` directory and run `python make_predictions.py`.

The code uses an ensemble of 5 category-specific models for prediction. As output, you will find the predictions by each individual model, as well as classifications based on majority votes for all provided text entries. Each category will output a separate file.

## Model performance

![data_performance_combined](https://github.com/AronSchulman/DTI-reviewer/assets/63584295/3366cd95-63dc-43ee-9974-6380f24e350d)

**A**: The composition of training data for each document classifier, balanced to contain equal proportions of positive and negative documents as shown in green and red, respectively. **B**: The document classifiers were trained with a stratified five-fold cross-validation. Thus, for each document category, we train and evaluate five binary classifiers. The blue and orange bars show the average accuracies and binary F1-scores derived from the five classifiers, respectively, and the black vertical lines indicate the standard errors of the mean (SEM). Note: the y-axis starts from 0.80 to better show the differences between the 16 classification tasks.

## Hyperparameters

![hyperparams](https://github.com/AronSchulman/DTI-reviewer/assets/63584295/84a8c6fc-0100-476a-b8ce-4e9c7d67059b)

The search space and optimal hyperparameters for each binary classifier after 20 iterations of random search. Each iteration was 10 epochs in duration. The hyperparameter search space was kept the same for all the categories.

## Extracting metadata from PubMed

We have provided the script `extract_abstracts_from_pubmed.py` for fetching article metadata. All you need is to provide a comma-separated string of PubMed IDs in the `config.yaml` file. For each article, the script will fetch the title + abstract concatenated into one string, the publishing journal, and the publication year.

## Tuning and training your own models

You can optimize and train your own models with the scripts `tune_abstract_classifier.py` and `train_abstract_classifier.py`, respectively. Configure your tuning and training at `config.yaml`. At minimum, the input `.csv` files should contain two columns: `text` and `label`. Note: the data in `data/train` contain the name of the category (e.g. `Accuracy`) instead of `label`.

You will need to create a Huggingface account and an access token (https://huggingface.co/docs/hub/en/security-tokens) for training the models. Place the token into the `cache/token` file.

## References

[1] Gururangan, Suchin, et al. "Don't stop pretraining: Adapt language models to domains and tasks." arXiv preprint arXiv:2004.10964 (2020).
