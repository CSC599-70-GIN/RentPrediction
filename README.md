# RentPrediction

## Project Description

I applied data handling and modeling skills taught during my CSC 599.70 (Introduction to Data Science) course taught by Grant Long to predict asking rents and answer modeling questions about NYC StreetEasy rental listings in a team of 3. 

The provided dataset includes a random sample of 12,000 homes posted during May - July 2018. The data includes features such as square footage, bedroom count, bathroom count, and included amenities.

The goal of the project is to minimize the mean squared error of estimated rents for a random set of listings from StreetEasy posted during August 2018. 

## External Datasets

My team and I incorporated two additional datasets to improve the accuracy of our predictions. 

The two datasets we chose were:
- [NYPD Complaint Dataset](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i)
- [311 Service Requests](https://nycopendata.socrata.com/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)

We chose these two datasets because we believe there is a correlation between the number of complaints to police and service requests and rental prices. Typically the greater the number or severity of crimes in a neighborhood, the less people are willing to pay to live there.

## Approach

Our approach was to test out multiple estimators with no hyperparameter tuning due to the time it takes to train and tune properly. 

Ultimately, our efforts resulted in using a gradient boosting model on viable features. We then hypertuned the parameters of our estimator to optimize the model’s performance using `sklearn`'s `RandomSearchCV`.

## Prerequisites

### Libraries and Modules Used 
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/docs/)
- [matplotlib](https://matplotlib.org/3.3.3/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)

### Python Environment
```
# create virtual environment
python3 -m venv env

# activate environment
source env/bin/activate

# install packages
pip install -r requirements.txt

# deactivate environemnt
deactivate
```

## Meta

Ishraq Khan – Contact me @ LinkedIn

Please comply with academic standards of honesty and do not pass off any of my work as your own. I am more than happy to explain if you need help understanding anything in this repository.

Distributed under the GPL license. See LICENSE for more information.
