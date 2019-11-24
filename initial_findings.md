**Explanation of model in terms of mean squared error and key features driving the model’s performance:**

Our mean squared error comes down to approximately 1.1 million while the number we needed to beat was approximately 4.2 million. When we ran 5 fold cross validation on our model our mean squared error ranged from 0.7 million to 1.6 million so we expect our model to at the very least beat the mean squared error by 2.5 million. The key features driving our model are the size_sqft, bathrooms, bbl, bedrooms, addr_zip, and bin. The size_sqft accounts for 33% of our model’s performance, bathrooms account for 19%, bbl accounts for 9%, bedrooms account for 8%, addr_zip accounts for 6%, and bin also accounts for 6%. We found out that these features by running a random forest regressor on our data and optimizing its hyperparameters. Then we looked at the feature importance to find out how much each feature contributes to our model.



**Intended strategy to improve performance for the final round:**

Our intended strategy consists of two parts: 1) testing and improving our model of choice, and 2) incorporating two external datasets. For this initial submission, we had some trouble with the hyperparameters so we will look into how to select good ranges for our hyperparameters in our current models and any other models (such as gradient boosting or ensemble learning) that we decide to test. We will also be adding on features based on the NYPD Historic Complaint Dataset and NYC Airbnb Open Dataset (linked below). Two ideas we have are aggregating the number of complaints per zip code or borough for the complaint dataset and finding the average price of an Airbnb room / listing by neighborhood.

Complaints: https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i

Airbnb: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
