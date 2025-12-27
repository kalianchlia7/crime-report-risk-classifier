import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split #Purpose: Splits your dataset into training and testing sets.
from sklearn.feature_extraction.text import TfidfVectorizer #Purpose: Converts text into numerical features the model can understand.
from sklearn.linear_model import LogisticRegression #Purpose: Classifier that predicts labels based on input features.
from sklearn.pipeline import Pipeline #Purpose: Chains multiple steps (like TF-IDF + classifier) into one object.
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

#LOAD AND CLEAN DATA 

df = pd.read_csv("Crime_Data_from_2020_to_Present.csv", nrows=1000)
#only reads first 1000 rows of file
#print(df.columns) <to pick columns>

text_column = "crime_description"
label_column = "area_name"

#drop null columns, keep only columns I need
df = df[[text_column, label_column]].dropna()
#creates new temporary dataframe that contains only two columns
#assigns new cleaned df back to variable df

#maek sure text is string
df[text_column] = df[text_column].astype(str)
#not checking label_column bc its already categorical text

#print(df.head()) 
       #shows first 5 rows of your DataFrame (df) by default
#print(df[label_column].value_counts()) 
       #Counts how many times each label occurs in your dataset
#sanity check


#BALANCE CLASSES

#AFTER CHECKING! Decided to balance.. heavily skewed to 'Central' : 794 vs. 'Topanga' : 1

#import resample

#find max class size
max_size = df[label_column].value_counts().max()

#upsample each class to match largest class
#groupby splits df into seperate jars by labeland 
df_balanced = df.groupby(label_column, group_keys=False).apply(
    lambda x: resample(x,
                       replace=True, #allows repeating rows
                       n_samples=max_size, #match target class
                       random_state=42) 

).sample(frac=1, random_state=42).reset_index(drop=True)

#shuffles all rows randomly
#drop=True discards old index and resets index to 0,1,2

#print(df_balanced[label_column].value_counts())
        #checking new class distribution



#SPLIT INTO TRAIN/TEST

X = df_balanced[text_column] #input text
y = df_balanced[label_column] #target label

#using the balanced df
#X model sees crime descriptions, y model predicts area_name
#scikit-learn expects features and labels as seperate variables

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, #20% of data goes to testing
    random_state=42, 
    stratify=y #keeps class proportions equal in train/test
)


#BUILT PIPELINE

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    #english: ignores common words like “the” or “and”
    #max features = only keeps 5000 most important words
    #outputs readable matrix of numbers for lr
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    #learns to predict area absed on the TF-IDF features
])


#TRAIN MODEL

pipeline.fit(X_train, y_train)
#each row in X-train (crim desc.) converted into numeric vector TF-DF score
#LR model learns weights for each word to predict the correct area_name

#print(len(pipeline.named_steps['tfidf'].get_feature_names_out()))
    #see how many features TF-IDF created
    #how many unique words model is using

#predict on test set
y_pred = pipeline.predict(X_test)

#X_test is never seen by the model during training
#y_pred contains the predicted area for each crime description in the test set


#EVALUATE

# Overall accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
#% of correct predictions overall

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#shows which areas get misclassified as which

# Detailed classification report
print(classification_report(y_test, y_pred))
#precision, recall, F1-score per area


#SAVE MODEL

#import joblib

#save pipeline to a file
joblib.dump(pipeline, "crime_are_classifier.pkl")
print("Model saved successfully!")
