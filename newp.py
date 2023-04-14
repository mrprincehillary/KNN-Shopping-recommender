from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process
import pandas as pd

# Load the dataset
df = pd.read_csv('electronics.csv')

#checking null values
df.isna().any()

# Remove duplicate records
df = df.drop_duplicates()

# Drop columns that are not needed
cols_to_drop = ['user_attr', 'split', 'timestamp', 'model_attr', 'brand', 'year']
df = df.drop(columns=cols_to_drop)
# Filter the dataset
df_filtered = df[df['item_id'] <= 100]
df_filtered = df_filtered.drop_duplicates(['item_id', 'user_id', 'rating'])
item_users = df_filtered.pivot_table(index='item_id', columns='user_id', values='rating',aggfunc='mean').fillna(0)
item_users_filtered = item_users[item_users.columns[item_users.notna().sum() >= 10]]
mat_item_users = csr_matrix(item_users_filtered.values)
 
#i have repetend the above code after installing scipy
#from scipy.sparse import csr_matrixs

# filter rows where the `item_id` is less than or equal to 100
#df_filtered = df[df['item_id'] <= 100]

# pivot the filtered DataFrame using `pivot_table()` and aggregate with `mean()`
item_users = df_filtered.pivot_table(index='item_id', columns='category', values='rating'), #aggfunc='mean')

# filter columns where the number of non-null values is greater than or equal to 10
#item_users_filtered = item_users[item_users.columns[item_users.notna().sum() >= 10]]

# create a sparse matrix from the filtered pivoted table
mat_item_users = csr_matrix(item_users_filtered.values)


# Create the KNN model
model_knn = NearestNeighbors(metric='manhattan', algorithm='brute', n_neighbors=20)
model_knn.fit(mat_item_users)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

# Define the KNN model with hyperparameters
model_knn_cv = KNeighborsRegressor(n_neighbors=20, metric='manhattan')

# Define the target variable
y = item_users_filtered.index.values

# Reshape y to a 2D array
y = y.reshape(-1, 1)

# Define the number of folds
n_folds = 5

# Evaluate the model using cross-validation
mse_scores = -cross_val_score(model_knn_cv, mat_item_users, y, cv=n_folds, scoring='neg_mean_squared_error')
rmse_scores = -cross_val_score(model_knn_cv, mat_item_users, y, cv=n_folds, scoring='neg_root_mean_squared_error')
mae_scores = -cross_val_score(model_knn_cv, mat_item_users, y, cv=n_folds, scoring='neg_mean_absolute_error')

if (y == 0).any():
    mse_percentage = 0
    rmse_percentage = 0
    mae_percentage = 0
else:
    mse_percentage = (1 - mse_scores  / y) * 100
    rmse_percentage = (1 - rmse_scores / y) * 100
    mae_percentage = (1 - mae_scores / y) * 100
    

accuracy = 100 - mae_scores.mean()

# Print the average performance metrics
print(f"Mean Squared Error: {mse_scores.mean():.4f}")
print(f"Root Mean Squared Error: {rmse_scores.mean():.4f}")
print(f"Mean Absolute Error: {mae_scores.mean():.4f}")
print(f"Accuracy: {accuracy:.2f}%")


# Create the Flask app
app = Flask(__name__,template_folder='.')

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the recommender page
@app.route('/recommender', methods=['POST'])
def recommender():
    item_name = request.form['item_name']
    n_recommendations = int(request.form['n_recommendations'])
    idx = process.extractOne(item_name, df['category'])[2]
    distances, indices = model_knn.kneighbors(mat_item_users[idx], n_neighbors=n_recommendations+1)
    recommendations = []
    for i in range(len(distances.flatten())):
        #for i in indices.flatten():
        if i != idx:
            recommendations.append((df['category'][i], round(distances.flatten()[i], 2)))
    return render_template('index.html', item_name=item_name, recommendations=recommendations)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
