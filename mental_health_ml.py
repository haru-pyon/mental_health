#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


# OSMH/OSMI Mental Health in Tech Survey 2016
# https://osmhhelp.org/research.html

df = pd.read_csv("mental_heath_2016.csv")


# # Data Cleaning

# In[5]:


# List of columns to drop
columns_to_drop = ['Are you self-employed?', 
                   "Is your primary role within your company related to tech/IT?",
                   'Do you know the options for mental health care available under your employer-provided coverage?',
                   "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:",
                   "Do you think that discussing a mental health disorder with your employer would have negative consequences?",
                   "Do you think that discussing a physical health issue with your employer would have negative consequences?",
                   "Would you feel comfortable discussing a mental health disorder with your coworkers?",
                   "Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?",
                   "Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?",
                   "Do you know local or online resources to seek help for a mental health disorder?",
                   "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?",
                   "If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?",
                   "If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?",
                   "If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?",
                   "Do you believe your productivity is ever affected by a mental health issue?",
                   "If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?",
                   "Were you aware of the options for mental health care provided by your previous employers?",
                   "Would you be willing to bring up a physical health issue with a potential employer in an interview?",
                   "Why or why not?",
                   "Would you bring up a mental health issue with a potential employer in an interview?",
                   "Why or why not?.1",
                   "Do you feel that being identified as a person with a mental health issue would hurt your career?",
                   "Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?",
                   "Do you have a family history of mental illness?",
                   "Have you had a mental health disorder in the past?",
                   "Do you currently have a mental health disorder?",
                   "If yes, what condition(s) have you been diagnosed with?",
                   "If maybe, what condition(s) do you believe you have?",
                   "Have you been diagnosed with a mental health condition by a medical professional?",
                   "If so, what condition(s) were you diagnosed with?",
                   "Have you ever sought treatment for a mental health issue from a mental health professional?",
                   "If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?",
                   "If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?",
                   "What country do you live in?",
                   "What US state or territory do you live in?",
                   "What country do you work in?",
                   "What US state or territory do you work in?",
                   "How willing would you be to share with friends and family that you have a mental illness?",
                   "If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:",
                   "What is your age?",
                   "What is your gender?"
                  ]

# Drop the specified columns
drop_df = df.drop(columns=columns_to_drop)


# In[8]:


#  Inspection of remaining columns
def unique_values(df, max_unique):
    for col in df.columns:
        if df[col].nunique() < max_unique:
            print(f"{col} has {df[col].nunique()} unique values \n{df[col].unique()}\n")
        else:
            print(f"{col} has {df[col].nunique()} unique values \n")
unique_values(drop_df, 20)


# In[9]:


# Categorizing employee position
def categorize_position(position):
    if position not in ["Back-end Developer", "Front-end Developer",
                       "Supervisor/Team Lead", "Back-end Developer|Front-end Developer"]:
        return "Other"
    else:
        return position
    
drop_df['Which of the following best describes your work position?'] = drop_df['Which of the following best describes your work position?'].apply(categorize_position)


# # Target value

# In[10]:


# Target Value inspection and categorization
drop_df["Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"].value_counts()


# In[11]:


def categorize_target(target):
    if target in ["Yes, I think they would", "Maybe", "Yes, they do"]:
        return 1
    else:
        return 0

drop_df['target'] = drop_df["Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"].apply(categorize_target)
drop_df = drop_df.drop(["Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?"], axis=1)


# # Train Test Split

# In[13]:


target = "target"
from sklearn.model_selection import train_test_split

X = drop_df.drop(target, axis = 1)
y = drop_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42,
                                                   test_size = 0.3)


# #  Set baseline

# In[14]:


y_train.value_counts(normalize=True) 


# In[15]:


from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
dummy.fit(X_train, y_train)


# In[16]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

y_true = y_train.copy()
y_pred = dummy.predict(X_train)
baseline = accuracy_score(y_true, y_pred)
print(f"The baseline to beat is {baseline}")


# # Build pipeline

# In[17]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# preprocess num columns
num_cols = X.select_dtypes("number").columns
num_trans = make_pipeline(SimpleImputer(strategy="median"),
                         StandardScaler())

cat_cols = X.select_dtypes("object").columns     
cat_trans = make_pipeline(SimpleImputer(strategy="constant", fill_value = "unknown"),
                         OneHotEncoder(handle_unknown="ignore", drop="if_binary"))
                          
preprocessor = make_column_transformer((num_trans, num_cols),
                                      (cat_trans, cat_cols))


# # Model selection

# In[18]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

classifiers = [RandomForestClassifier(), SVC(), LogisticRegression(),
              DecisionTreeClassifier()]

for clf in classifiers:
    pipe = make_pipeline(preprocessor, clf)
    grid = GridSearchCV(pipe, cv = 10, scoring = "recall", param_grid ={})
    grid.fit(X_train, y_train)
    score = grid.best_score_
    print(f"Classifier {clf} scored {score}\n")


# # Tune SVC

# In[19]:


param_grid = {"svc__C": [0.1, 1, 2, 1.5], "svc__kernel": ["rbf", "sigmoid"],
             "svc__gamma": ["scale", "auto"]}

pipe = make_pipeline(preprocessor, SVC())
grid = GridSearchCV(pipe, param_grid = param_grid, scoring="accuracy", cv = 10)
grid.fit(X_train, y_train)


# Make predictions on the test data
y_pred = grid.predict(X_train)

# Calculate accuracy score
accuracy = accuracy_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)

print("Best score:", grid.best_score_)

print("Accuracy: ", accuracy)
print("Recall: ", recall)
print("Precision: ", precision)
print("Best score:", grid.best_params_)


# In[20]:


import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
pred = grid.predict(X_train)
cm = confusion_matrix(y_train, pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap='cividis')
plt.show()


# # Final Evaluation

# In[21]:


# Make predictions on the test data
y_test_pred = grid.predict(X_test)

recall = recall_score(y_test, y_test_pred)
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)

print("Accuracy:", accuracy)
print("Recall: ", recall)
print("Precision:", precision)


# In[22]:


cm = confusion_matrix(y_test, y_test_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap='cividis')
# plt.savefig('test.png', transparent=True, bbox_inches='tight')
plt.show()


# # Post Test Analyzation

# In[23]:


test_df = drop_df.copy()


# In[24]:


test_df.columns = ['employees size',
       'primarily tech',
       'provide benefits',
       'discuss',
       'resources',
       'anonymity protected',
       'seriously as physical',
       'heard negative',
       'previous employers',
       'provided benefits(p)',
       'discuss(p)',
       'resources(p)',
       'anonymity protected(p)',
       'discuss mental(p)',
       'discuss physical(p)',
       'discuss co-workers(p)',
       'discuss boss(p)?',
       'seriously as physical(p)',
       'heard negative(p)',
       'unsupportive',
       'position?',
       'remote wark','negative view']


# In[25]:


# dummies:
test_df = pd.get_dummies(data = test_df, columns = test_df.columns,
                  drop_first = True)


# In[26]:


# Define the columns to convert
columns_to_convert = test_df.columns
# Convert 'True' and 'False' to 1 and 0 in the specified columns
test_df[columns_to_convert] = test_df[columns_to_convert].replace({True: 1, False: 0})


# In[27]:


import seaborn as sns

def abbreviate_column_names(column_names):
    return [name[:20] for name in column_names]

plt.figure(figsize=(20, 10))

correlation_matrix = test_df.corr(numeric_only=True)


abbreviated_column_names = abbreviate_column_names(correlation_matrix.columns)


sns.heatmap(cmap = "rocket", data = correlation_matrix, fmt=".1f", annot=False, xticklabels=abbreviated_column_names, yticklabels=abbreviated_column_names)

plt.show()


# In[29]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


correlation_matrix = test_df.corr()

# Sort the correlations of the target column in descending order
correlations_with_target = correlation_matrix["negative view_1"].sort_values(ascending=False)

# Select the top 5 most correlated columns (excluding the target column itself)
top_5_correlated_columns = correlations_with_target[1:6]

# Select the top 5 non-correlated columns
top_5_nocorrelated_columns = correlations_with_target[-5:]

# Concatenate the top correlated columns and non-correlated columns
vis_column = pd.concat([top_5_correlated_columns, top_5_nocorrelated_columns], axis=0)

# Create a bar chart to visualize the top correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=vis_column.values, y=vis_column.index, palette="viridis",
           edgecolor = "black")
plt.xlabel("Correlation", fontsize = 15)
plt.ylabel("")
plt.title(f"Top Correlations with Negative View", fontsize = 20)
plt.yticks(fontsize=15, rotation=0)
plt.xticks(fontsize=15, rotation=0)

plt.show()


# In[30]:


from sklearn.dummy import DummyClassifier

dummy = DummyClassifier()
dummy.fit(X_test, y_test)


# In[31]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

y_true = y_test.copy()
y_pred = dummy.predict(X_test)
baseline = accuracy_score(y_true, y_pred)
print(f"The baseline to beat is {baseline}")

