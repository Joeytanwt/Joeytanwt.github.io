---
layout: post
author: Joey Tan
title: "Airbnb Business Analytics"
categories: ITD214
---
# Project Background
The objective of this project is to build a sentiment classifier to analyse guest feedback for a specific Airbnb property. Due to the large volume of data in the complete Singapore Airbnb dataset, this analysis focuses on **Listing ID 42081657**, a property with approximately 500 reviews.

## Data Sources
Two distinct datasets were used for this project: a target dataset for analysis and a labelled training dataset for model training.

**1. Target Data (Airbnb Reviews)**
*  Source: [InsideAirbnb](https://insideairbnb.com/get-the-data/)
* The initial dataset contains 38,350 reviews across 6 columns: ```listing_id```, ```id```, ```date```, ```reviewer_id```, ```reviewer_name```, and ```comments```.
    
![alt text](/assets/images/df_sample.jpg)

**2. Training Data (Trip Advisor Hotel Reviews)**
* Source: [Trip Advisor (via Kaggle)](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) 
* Contains 20,491 rows and 2 columns: ```Review``` and ```Rating```.

![alt text](/assets/images/train_df_sample.jpg)

### Data Preparation
#### **Airbnb Data**
The primary focus of the cleaning process was the ```comments``` column, which contained dirty data:

* HTML tags such as ```<br/>```

![alt text](/assets/images/html_tags.jpg)

* Non-English text: 

![alt text](/assets/images/non_engsample.jpg)

* Unnecessary columns (```listing_id```, ```id```, ```date```, etc.) were dropped.

* **Nulls** and **Duplicates** were removed.

After cleaning, the final dataset consisted of 370 unique reviews.

#### **Trip Advisor Data**
As the Airbnb dataset didn't contain sentiment labels, an external Trip Advisor Hotel Reviews dataset was used to train the classifier. The dataset contained a ```Rating``` column with values ranging from 1 to 5. To convert this into a supervised learning problem for sentiment analysis, the ratings were transformed into a categorical label with the following logic:

* **Negative (0):** Ratings of 1 and 2.
* **Neutral (1):** Rating of 3.
* **Positive (2):** Ratings of 4 and 5.

The training data was then processed using the same cleaning steps above and balanced by sampling 500 rows of each class.

![alt text](/assets/images/balanced(with_neutral).jpg)

---
# Modelling
## Feature Extraction with DistilBERT
This project uses **Transfer Learning** by leveraging a pre-trained DistilBERT model. Unlike traditional Bag-of-Words or TF-IDF methods that treat words independently, DistilBERT processes text bidirectionally to understand the context and semantics of sentences. This is highly beneficial for interpreting the informal language found in hotel reviews.
```
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel,
                                                    ppb.DistilBertTokenizer,
                                                    'distilbert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
```
Instead of fine-tuning the entire transformer architecture, the feature extraction approach was used:

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

  # embedding
  embeddings = []
  with torch.no_grad():
    for batch, batch_mask in dataloader:
      outputs = model(batch, attention_mask=batch_mask)
      batch_output = outputs[0][:, 0, :].numpy() # extraxt CLS token only
      embeddings.append(batch_output)

  x = np.concatenate(embeddings, axis=0)
```
## Classification
Once the DistilBERT features were extracted, they served as the input for 3 classifiers that performed the final multi-class classification task to categorise reviews into the 3 labels (Negative, Neutral, or Positive). The models were then evaluated using a train-test split (80/20).

```
def evaluate_model(mod, x_train, y_train, x_test, y_test):
  mod.fit(x_train, y_train)
  y_pred = mod.predict(x_test)

  return accuracy_score(y_test, y_pred)

lr = LogisticRegression(multi_class='multinomial')
svm = SVC(probability=True)
tree = DecisionTreeClassifier()

models = {'Logistic Regression':lr, 'SVM':svm, 'Decision Tree':tree}
scores = []
for model_name, mod in models.items():
  score = evaluate_model(mod, x_train, y_train, x_test, y_test)
  scores.append({'Model': model_name, 'Score': score})

df_scores = pd.DataFrame(scores)
df_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)
```

## Evaluation

|Model|Score|
|---|---|
| SVM | 0.72 |
| Logistic Regression | 0.69 |
| Decision Tree | 0.50 |

The SVM model was chosen as it achieved the highest accuracy of 72%. However, it struggled with the "Neutral" class (0.65 recall), misclassifying 35% of all neutral sentiments as negative/positive.

![alt text](/assets/images/svm_class_report.jpg)
![alt text](/assets/images/svm_cm.jpg)

The ROC curve also showed that the model was overfitting slightly (0.87 train AUC vs 0.79 test AUC).

![alt text](/assets/images/svm_roc.jpg)

## Tuning Model

To resolve this, a Grid Search was performed with the following parameters:

* Log-scale regularization (C): Used a log scale to explore stronger regularization values.

* Macro-F1 scoring metric: The evaluation metric was changed from the default accuracy to f1_macro, which equally weights performance across all three classes.

```
params = [
    {'kernel': ['linear'],
     'C': np.logspace(-10, 1, 10)},
    {'kernel': ['rbf'],
     'C': np.logspace(-10, 1, 10),
     'gamma': ['scale', 'auto', 0.1, 0.01]}
]
grid_search = GridSearchCV(svm, params, scoring='f1_macro')
grid_search.fit(x_train, y_train)
```

After tuning, the model improved slightly across all metrics, though it still struggles with the neutral class. This is likely due to the ambiguity in neutral reviews.

![alt text](/assets/images/svm_tuned_class_report.jpg)
![alt text](/assets/images/svm_tuned_cm.jpg)
![alt text](/assets/images/svm_tuned_roc.jpg)

---
# Analysis and Recommendations

The trained model was applied to the 370 cleaned reviews of Airbnb Listing 42081657. The predicted sentiment distribution is as follows:

|Sentiment|Count|Proportion|
|---|---|---|
|Positive | 220 | 59% |
|Neutral | 139 | 38% |
|Negative | 11 | 3% |

The overall guest perception is positive, though some operational friction points exist.

### Favorable Guest Satisfaction (59% Positive)

![alt text](/assets/images/pos_sample_10.jpg)

Guests frequently praise the location ("conveniently located near the MTR, with plenty of restaurants and shops in the area") and the high level of service provided by the staff ("very quick responses from the hosts"). This indicates that the property's personnel and geographic convenience are its strongest assets.

### High Volume of Neutral Sentiments (38%)

![alt text](/assets/images/neutral_sample_10.jpg)

Neutral reviews frequently contain mixed sentiments (e.g., praising the location but mentioning a minor inconvenience) or purely factual statements without strong emotional vocabulary. A number of reviews seem to be misclassified, such as "Clean and quiet room. Good location. Friendly checkin and checkout. Clean kitchen." (Positive) and "Totally bad with the construction outside. Noisy, dangerous and hard to go in & out as the only way in is through the back lane. Noise was loud during day time and also on Sunday morning" (Negative).

### Low Attrition Rate (3% Negative) with Specific Pain Points

![alt text](/assets/images/neg_revs.jpg)

Only 11 out of 370 reviews were flagged as negative. While this low volume is excellent, isolating these 11 reviews reveals recurring issues that disrupt the guest experience:

 * **External Noise:** Multiple guests complained about noise from adjacent construction sites ("noisy skyscraper construction all night").

* **Internal Noise:** Some guests reported disruptive noises from the elevator ("elevator bangs and slams"). 

* **Bathroom**: A few reviews highlighted complaints regarding bathroom maintenance (e.g., sliding doors not locking, sewer smells and no hot water).

## Recommendations for the Property Owner

**1. Manage Guest Expectations**

* Update Listing Description: Since construction noise is a recurring external issue beyond the owner's control, proactively stating this in the Airbnb listing will manage guest expectations before they book.

* Provide Comfort Amenities: To mitigate the construction and elevator noise, the owner could provide complimentary earplugs in every room.

**2.  Prioritize Targeted Maintenance**

* Some reviews point to specific infrastructure issues (elevator noise, plumbing/sewer smells, broken AC, and faulty bathroom locks). The owner should identify which specific room numbers correlate with these complaints and temporarily block them from being booked until plumbing and soundproofing repairs are completed.

* Improve Shower Facilities: Ensure hot water heaters are consistently serviced and that bathroom ventilation is improved to resolve the lingering odor issues mentioned.

**3. Elevate the "Neutral" Experience**

* Enhance the Check-in Process: One review noted the check-in was "overly complicated." Streamlining this process (e.g., using a clear visual guide, a digital lockbox, or a pre-arrival WhatsApp message) will provide a good first impression.

* Leverage Staff Strengths: Since the staff is frequently praised, empower them to add personalized touches (e.g., a handwritten welcome note or local dining recommendations). These low-cost, high-impact gestures are proven to convert "neutral" stays into positive reviews.

## Model Limitations

While the model shows a reasonable accuracy of 0.74, there are limitations that need to be considered:

**1. Misclassification of Negative Reviews:** During evaluation, the model misclassified 21 out of 100 negative reviews. This indicates a blind spot where critical feedback might be overlooked as actual negative experiences are downplayed to a neutral sentiment.

**2. Struggles with Neutral Sentiment:** The model shows challenges with the neutral class (0.68 recall). This means that a significant portion (32%) of actual neutral reviews are being misclassified. This, however, acts as a safety net, as it is better to misclassify neutral reviews as negative because the business cost of ignoring a guest complaint is much higher than the cost of over-investigating a neutral comment. 

---
# AI Ethics

## Privacy and Data Anonymization

The project took steps to anonymize the data by dropping explicit personally identifiable columns, such as ```reviewer_id``` and ```reviewer_name```.

However, a residual privacy risk remains within the unstructured text column. Guests frequently include names of hosts or travel companions in their reviews (e.g., "Thank you Nigel"). While this dataset is public, deploying the model in a private enterprise would require additional processing steps, such as using Named Entity Recognition (NER) to mask names and sensitive details before the data is stored or processed.

## Fairness and Representation Bias

The data cleaning process intentionally filtered out non-English reviews to simplify the project, this decision introduces representation bias. By excluding non-English feedback, the model silences the voices of international guests from non-English-speaking regions.

Additionally, training the model on a general Trip Advisor dataset introduces a domain and cultural shift. The model may struggle to accurately evaluate Singlish or localized terms found in the Singaporean Airbnb reviews. A fair model would require a diverse, multilingual training dataset that accurately reflects the property's global guest demographic.

## Accuracy and Labeling Assumptions

The project relied on proxy labeling, assuming that a 3-star Trip Advisor rating equates to a "Neutral" sentiment. In reality, a 3-star review may contain highly polarized statements (e.g., "The location was amazing, but the room was filthy").

Because the model struggles most with this neutral class, there is a risk of misclassifying actionable negative feedback as neutral. If severe complaints—such as safety concerns or hygiene issues—are absorbed into the neutral category, the property owner might fail to address critical problems.

## Accountability

The property owners must be aware of the model's limitations (70% accuracy rate and 68% recall in neutral class) and accountable for how they interpret insights. High-stakes decisions, such as firing a cleaning vendor based on "negative" sentiment trends, should always involve human-in-the-loop verification.

## Transparency and Explainability

DistilBERT generates high-dimensional contextual embeddings by processing text with millions of parameters and attention mechanisms, while SVM classifiers draw complex decision boundaries in multidimensional spaces. It is difficult to pinpoint exactly which words or phrases triggered a particular sentiment prediction. Future iterations of this project may include explainable AI (XAI) tools to highlight the specific keywords influencing the model's decisions.

---
# Source Codes and Datasets
[Notebook](https://github.com/Joeytanwt/Sentiment_Classification_with_DistilBERT)

[Aribnb Data](https://data.insideairbnb.com/singapore/sg/singapore/2025-09-28/data/reviews.csv.gz)

[Trip Advisor Data](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)