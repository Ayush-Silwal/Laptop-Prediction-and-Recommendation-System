#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df =pd.read_csv('laptop_data.csv')


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df.isnull().sum()


# In[6]:


df.duplicated().sum()


# In[7]:


duplicated_rows = df[df.duplicated(keep=False)]
print(duplicated_rows)


# In[8]:


df.drop(columns=["Unnamed: 0"],inplace=True)


# In[9]:


df.head()


# In[10]:


df["Ram"]=df["Ram"].str.replace("GB","")


# In[11]:


df["Weight"]=df["Weight"].str.replace("kg","")


# In[12]:


df["Ram"]=df["Ram"].astype("int")


# In[13]:


df["Weight"]=df["Weight"].astype("float")


# In[14]:


df.info()


# In[15]:


df["ScreenResolution"].value_counts()


# In[ ]:





# In[16]:


sns.displot(df['Price'])


# In[17]:


sns.barplot(x=df["Company"],y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()


# In[18]:


df["Company"].value_counts().plot(kind="bar")


# In[19]:


df["TypeName"].value_counts().plot(kind="bar")


# In[20]:


sns.barplot(x=df["TypeName"],y=df["Price"])
plt.xticks(rotation="vertical")
plt.show()


# In[21]:


sns.displot(df["Inches"])


# In[22]:


sns.scatterplot(x="Inches",y="Price",data=df)


# In[23]:


df.head()


# In[24]:


df["Touchscreen"]=df["ScreenResolution"].apply(lambda x:1 if "Touchscreen" in x else 0)


# In[25]:


df.head()


# In[26]:


df["Touchscreen"].value_counts().plot(kind="bar")
plt.xticks(rotation="horizontal")
plt.show()


# In[27]:


sns.barplot(x="Touchscreen",y="Price",data=df)


# In[28]:


df["Ips"]=df["ScreenResolution"].apply(lambda x:1 if "IPS" in x else 0)


# In[29]:


df.head()


# In[30]:


sns.barplot(x=df["Ips"],y=df["Price"])


# In[31]:


df["ScreenResolution"].str.split("x")


# In[32]:


temp= df["ScreenResolution"].str.split("x",n=1,expand=True)


# In[33]:


df["X_res"]=temp[0]
df["Y_res"]=temp[1]


# In[34]:


df.head()


# In[35]:


df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])


# In[36]:


df.head()


# In[37]:


df["X_res"] = df["X_res"].astype("int")
df["Y_res"] = df["Y_res"].astype("int")


# In[38]:


df.info()


# In[39]:


numeric_df = df.select_dtypes(include=['number'])
numeric_df.corr()['Price']


# In[40]:


df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')


# In[41]:


numeric_df = df.select_dtypes(include=['number'])
numeric_df.corr()['Price']


# In[42]:


df.drop(columns=["ScreenResolution","X_res","Y_res","Inches"],inplace=True)


# In[43]:


df.head()


# In[44]:


df["Cpu"].value_counts()


# In[45]:


df['Cpu Name'] = df['Cpu'].apply(lambda x:" ".join(x.split()[0:3]))


# In[46]:


df.head()


# In[47]:


def fetch_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'


# In[48]:


df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)


# In[49]:


df.head()


# In[50]:


df['Cpu brand'].value_counts().plot(kind='bar')


# In[51]:


sns.barplot(x=df['Cpu brand'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[52]:


df.drop(columns=['Cpu','Cpu Name'],inplace=True)


# In[53]:


df.head()


# In[54]:


df['Ram'].value_counts().plot(kind='bar')


# In[55]:


sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[56]:


df['Memory'].value_counts()


# In[57]:


df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')
new = df["Memory"].str.split("+", n=1, expand=True)

df["first"] = new[0].str.strip()
df["second"] = new[1]

df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['first'] = df['first'].str.replace(r'\D', '', regex=True)

# Use .loc[] to fill NaN values to avoid the warning
df.loc[:, "second"] = df["second"].fillna("0")

df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)

df['second'] = df['second'].str.replace(r'\D', '', regex=True)

df["first"] = df["first"].astype(int)
df["second"] = df["second"].astype(int)

df["HDD"] = (df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"])
df["SSD"] = (df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"])
df["Hybrid"] = (df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"])
df["Flash_Storage"] = (df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"])

df.drop(columns=['first', 'second', 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid',
                 'Layer1Flash_Storage', 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid',
                 'Layer2Flash_Storage'], inplace=True)


# In[58]:


df.head()


# In[59]:


df.drop(columns=['Memory'],inplace=True)


# In[60]:


numeric_df = df.select_dtypes(include=['number'])
numeric_df.corr()['Price']


# In[61]:


df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)


# In[62]:


df.head()


# In[63]:


df['Gpu'].value_counts()


# In[64]:


df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])


# In[65]:


df.head()


# In[66]:


df['Gpu brand'].value_counts()


# In[67]:


df = df[df['Gpu brand'] != 'ARM']


# In[68]:


df['Gpu brand'].value_counts()


# In[69]:


sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()


# In[70]:


df.drop(columns=['Gpu'],inplace=True)


# In[71]:


df.head()


# In[72]:


df['OpSys'].value_counts()


# In[73]:


sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()


# In[74]:


def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'


# In[75]:


df['os'] = df['OpSys'].apply(cat_os)


# In[76]:


sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='horizontal')
plt.show()


# In[77]:


df.drop(columns=['OpSys'],inplace=True)


# In[78]:


df.head()


# In[79]:


sns.displot(df['Weight'])


# In[80]:


sns.scatterplot(x=df['Weight'],y=df['Price'])


# In[81]:


numeric_df.corr()['Price']


# In[82]:


df.info()


# In[83]:


sns.heatmap(numeric_df.corr())


# In[84]:


sns.displot(np.log(df['Price']))


# In[85]:


X = df.drop(columns=['Price'])
y = np.log(df['Price'])


# In[86]:


X


# In[87]:


y


# ###Training and Splitting data

# In[88]:


import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[89]:


# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape, X_test.shape


# In[90]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# Build a Decision Tree

# In[91]:


import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.tree_ = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        if num_samples <= 1 or (self.max_depth is not None and depth >= self.max_depth):
            return np.mean(y)

        best_split = self._find_best_split(X, y)
        if best_split is None:
            return np.mean(y)
        
        left_indices = X[:, best_split['feature']] <= best_split['value']
        right_indices = X[:, best_split['feature']] > best_split['value']
        
        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {'feature': best_split['feature'], 'value': best_split['value'], 'left': left_tree, 'right': right_tree}
    
    def _find_best_split(self, X, y):
        best_split = None
        best_mse = float('inf')
        num_features = X.shape[1]

        for feature in range(num_features):
            values = np.unique(X[:, feature])
            for value in values:
                left_indices = X[:, feature] <= value
                right_indices = X[:, feature] > value
                
                if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
                    continue
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                
                mse = (np.var(left_y) * len(left_y) + np.var(right_y) * len(right_y)) / len(y)
                
                if mse < best_mse:
                    best_split = {'feature': feature, 'value': value}
                    best_mse = mse
        
        return best_split

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._predict(sample, self.tree_) for sample in X])
    
    def _predict(self, sample, tree):
        if not isinstance(tree, dict):
            return tree
        
        if sample[tree['feature']] <= tree['value']:
            return self._predict(sample, tree['left'])
        else:
            return self._predict(sample, tree['right'])


# Build Random Forest

# In[92]:


from sklearn.utils import resample

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)


# In[93]:


# Define models
models = {
    "Decision Tree": DecisionTree(max_depth=5),
    "Random Forest": RandomForest(n_estimators=100, max_depth=10),
}


# In[94]:


# Example training and evaluation
for model_name, model in models.items():
    model.fit(X_train, y_train)  # Train model


# ###Train and Evaluate

# In[95]:


import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Function to evaluate model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square

# Example training and evaluation
results = []

for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate Train and Test dataset
    model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    # Print results
    print(f"{model_name}:")
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))
    print('----------------------------------')
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    print('='*35)
    print('\n')

    # Store results
    results.append({
        'model': model_name,
        'train_mae': model_train_mae,
        'train_rmse': model_train_rmse,
        'train_r2': model_train_r2,
        'test_mae': model_test_mae,
        'test_rmse': model_test_rmse,
        'test_r2': model_test_r2
    })

# Optional: Convert results to a DataFrame for easier analysis
import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)


# In[96]:


# Given R² scores
r2_scores = {
    'Decision Tree': {'train_r2': 0.839276, 'test_r2': 0.786249},
    'Random Forest Regressor': {'train_r2': 0.861095, 'test_r2': 0.801851}
}

# Calculate accuracy percentages
accuracy_percentages = {}
for model_name, scores in r2_scores.items():
    train_accuracy_percentage = scores['train_r2'] * 100
    test_accuracy_percentage = scores['test_r2'] * 100
    accuracy_percentages[model_name] = {
        'train_accuracy_percentage': train_accuracy_percentage,
        'test_accuracy_percentage': test_accuracy_percentage
    }

# Print accuracy percentages
for model_name, percentages in accuracy_percentages.items():
    print(f"{model_name}:")
    print(f"- Training Set Accuracy Percentage: {percentages['train_accuracy_percentage']:.2f}%")
    print(f"- Test Set Accuracy Percentage: {percentages['test_accuracy_percentage']:.2f}%")
    print("="*35)
    print('\n')


# In[97]:


import matplotlib.pyplot as plt

# Plot predictions
plt.scatter(y_test, y_test, color='red', label='Actual Prices')  # Actual prices in red
plt.scatter(y_test, y_test_pred, color='blue', label='Predicted Prices')  # Predicted prices in blue
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()


# In[98]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

pipe = Pipeline([
    ('step1', ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')),
    ('step2', RandomForest(n_estimators=100,  max_depth=15))
])

pipe.fit(X_train, y_train)





# In[99]:


import pickle
pickle.dump(df,open('df.pkl','wb'))
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[100]:


X.sample()


# In[101]:


import pandas as pd
import pickle
import numpy as np

# Load the trained model (pipeline with preprocessor)
with open('pipe.pkl', 'rb') as file:
    model = pickle.load(file)

# Example test data for laptop price prediction
test_data = pd.DataFrame({
    'Company': ['Asus'],
    'TypeName': ['Gaming'],
    'Ram': [16],
    'Weight': [2.5],
    'Touchscreen': [0],  # 1 if yes, 0 if no
    'Ips': [1],          # 1 if yes, 0 if no
    'ppi': [141.211998],     # Example value, compute based on resolution and screen size
    'Cpu brand': ['Intel Core i7'],
    'HDD': [0],
    'SSD': [0],
    'Gpu brand': ['Nvidia'],
    'os': ['Windows']
})

# Make predictions directly using the pipeline
predicted_price = model.predict(test_data)

# Assuming the target was log-transformed during training
final_price = int(np.exp(predicted_price[0]))

print(f'Predicted Price: {final_price}')


# In[102]:


# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score,mean_absolute_error


# In[103]:


# step1 = ColumnTransformer(transformers=[
#     ('col_tnf',OneHotEncoder(sparse_output=False,drop='first'),[0,1,7,10,11])
# ],remainder='passthrough')

# step2 = RandomForestRegressor(n_estimators=100,
#                               random_state=3,
#                               max_samples=0.5,
#                               max_features=0.75,
#                               max_depth=15)

# pipe = Pipeline([
#     ('step1',step1),
#     ('step2',step2)
# ])

# pipe.fit(X_train,y_train)

# y_pred = pipe.predict(X_test)

# print('R2 score',r2_score(y_test,y_pred))
# print('MAE',mean_absolute_error(y_test,y_pred))


# In[ ]:





# ### Exporting Model

# In[104]:


# import pickle
# pickle.dump(df,open('df.pkl','wb'))
# pickle.dump(pipe,open('pipe.pkl','wb'))


# In[105]:


df


# In[ ]:




