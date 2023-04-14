import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

import pandas as pd
df = pd.read_csv('electronics.csv')

#df.info()

#df.head

#checking for Null values

df.isna().any()

#from scipy.sparse import csr_matrix
# filter rows where the `item_id` is less than or equal to 100
#df_filtered = df[df['item_id'] <= 100]

# pivot the filtered DataFrame using `pivot_table()` and aggregate with `mean()`
#item_users = df_filtered.pivot_table(index='item_id', columns='user_id', values='rating', aggfunc='mean').fillna(0)

# filter columns where the number of non-null values is greater than or equal to 10
#item_users_filtered = item_users[item_users.columns[item_users.notna().sum() >= 10]]
#mat_item_users=csr_matrix(item_users_filtered .values)
#item_users 

#i have repetend the above code after installing scipy
from scipy.sparse import csr_matrix

# filter rows where the `item_id` is less than or equal to 100
df_filtered = df[df['item_id'] <= 100]

# pivot the filtered DataFrame using `pivot_table()` and aggregate with `mean()`
item_users = df_filtered.pivot_table(index='item_id', columns='category', values='rating'), #aggfunc='mean')

# filter columns where the number of non-null values is greater than or equal to 10
item_users_filtered = item_users[item_users.columns[item_users.notna().sum() >= 10]]

# create a sparse matrix from the filtered pivoted table
mat_item_users = csr_matrix(item_users_filtered.values)

#knn engine model

from sklearn.neighbors import NearestNeighbors
model_knn= NearestNeighbors(metric='cosine', algorithm='brute',n_neighbors=5)

#fitting our model in dataset

model_knn.fit(mat_item_users)

from fuzzywuzzy import process

def recommender(item_name):
    idx = process.extractOne(item_name, df['category'])
    print(idx)

recommender('Computers & Accessories')



def recommender(item_name,data,model,n_recommendations):
    model.fit(data)
    idx = process.extractOne(item_name, df['category'])[2]
    print('Iteim Selected: ',df['category'][idx],'Index: ',idx)
    print('Searching for recommendations.......')
    distances, indices=model.kneighbors(data[idx], n_neighbors=n_recommendations)
    for i in indices:
        print(df['category'][i].where(i!=idx))
        
recommender('phone',mat_item_users, model_knn,20)

#if __name__ == '__main__':
 #   app.run(debug=True)