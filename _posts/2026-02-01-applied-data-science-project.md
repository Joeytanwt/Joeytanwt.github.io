---
layout: post
author: Joey Tan
title: "Airbnb Business Analytics"
categories: ITD214
---
# Project Background
The objective of this project is to build a sentiment classifier to analyse guest feedback for a specific Airbnb property. Due to the large volume of data in the complete Singapore Airbnb dataset, this analysis focuses on Listing ID 42081657, a property with approximately 500 reviews.

## Data Sources
Two distinct datasets were used for this project: a target dataset for analysis and a labeled training dataset for model training.

1. Target Data (Airbnb Reviews)
    *  Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/)
    * The initial dataset contains 38,350 reviews across 6 columns: ```listing_id```, ```id```, ```date```, ```reviewer_id```, ```reviewer_name```, and ```comments```.
    ![alt text](df_sample.jpg)

2. Training Data (TripAdvisor Hotel Reviews)
    * Source: [TripAdvisor (via Kaggle)](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) 
    * Contains 20,491 rows and 2 columns: ```Review``` and ```Rating```.
    ![alt text](train_df_sample.jpg)

### Data Preparation
#### Airbnb Data
The primary focus of the cleaning process was the ```comments``` column, which contained dirty data:

* HTML tags such as ```<br/>```
![alt text](html_tags.jpg)

* Non-English text: 
![alt text](non_engsample.jpg)

* Unnecessary columns (listing_id, id, date, etc.) were dropped.

* Nulls and Duplicates were removed.

After cleaning, the final dataset consisted of 372 unique reviews.

#### TripAdvisor Data
As the Airbnb dataset didn't contain sentiment labels, an external TripAdvisor Hotel Reviews dataset was used to train the classifier. The dataset contained a ```Rating``` column with values ranging from 1 to 5. To convert this into a supervised learning problem for sentiment analysis, the ratings were transformed into a categorical label with the following logic:

* Negative (0): Ratings of 1 and 2.
* Neutral (1): Rating of 3.
* Positive (2): Ratings of 4 and 5.

The training data was then processed using the same cleaning steps above and balanced by sampling 500 rows of each class.

![alt text](balanced(with_neutral).jpg)

# Modelling
## Feature Extraction with DistilBERT
This project employed Transfer Learning by leveraging a pre-trained DistilBERT model.
```
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                    ppb.DistilBertTokenizer,
                                                    'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```
Instead of fine-tuning the entire transformer architecture, the Feature Extraction approach was used:

* The review text was passed through the DistilBERT tokenizer and model.

* The hidden states from the last layer (specifically the [CLS] token representation) were extracted to serve as numerical features (embeddings) representing the semantic meaning of each review.

```
  # tokenization
  tokenized = tokenizer(text,
                        padding=True, # padding & masking
                        return_tensors='pt')

  # batching
  dataset = TensorDataset(tokenized['input_ids'],
                          tokenized['attention_mask'])
  dataloader = DataLoader(dataset, # batching for efficiency
                          batch_size=32,
                          shuffle=False)

  #embedding
  embeddings = []
  with torch.no_grad():
    for batch, batch_mask in dataloader:
      outputs = model(batch, attention_mask=batch_mask)
      batch_output = outputs[0][:, 0, :].numpy() # extraxt CLS token only
      embeddings.append(batch_output)

  x = np.concatenate(embeddings, axis=0)
```
## Classification with Logistic Regression
Once the DistilBERT features were extracted, they served as the input for a Logistic Regression classifier which performed the final multi-class classification task to categorize reviews into the 3 labels (Negative, Neutral, or Positive).


### Evaluation
The model was evaluated using a train-test split (80/20).
#### Initial Performance
The baseline model achieved an accuracy of ~68% and struggled with the "Neutral" class, misclassifying almost half of all neutral sentiments as negative/positive.

![alt text](mod_1_met.jpg)
![alt text](mod_1_cm.jpg)

The ROC curve also showed that the model was overfitting with a high training AUC but a low validation AUC.

![alt text](mod_1_roc.jpg)

To resolve this, a Grid Search was performed with the following adjustments:

* Log-scale regularization (C): Switched from a linear scale to a log scale to explore stronger regularization values.

* L1 and L2 penalties: Both penalty types were included in the search. L1 encourages sparsity (zeroing out less important features), while L2 shrinks all weights uniformly. This gave the model more options to generalize.

* Macro-F1 scoring metric: The evaluation metric was changed from default accuracy to f1_macro, which equally weights performance across all three classes.

```
parameters = {
    'C': np.logspace(-10, 1, 10),
    'penalty': ['l1','l2']
    }

clf = LogisticRegression(multi_class='multinomial')

grid_search = GridSearchCV(clf, parameters, scoring='f1_macro')
grid_search.fit(x_train, y_train)
```
#### Tuned Performance
After tuning, the model improved slightly across all metrics, though it still struggles with the neutral class. This is likely due to the ambiguity in neutral reviews.

![alt text](mod_2_met.jpg)
![alt text](mod_2_cm.jpg)

The tuned model also showed a smaller gap on the ROC curve between training and test sets.

![alt text](mod_2_roc.jpg)

## Predicting Airbnb Sentiment
The trained model was applied to the 372 cleaned reviews of Airbnb Listing 42081657. The predicted sentiment distribution is as follows:

|Sentiment|Count|Proportion|
|---|---|---|
|Positive | 239 | 64% |
|Neutral | 113 | 30% |
|Negative | 21 | 6% |

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
