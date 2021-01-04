import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def joinRentNYPD(rentDf, nypdDf):
	"""
	Takes in the all NYC zipcodes in the rent dataset and returns the NYCPD precinct associated with it.
	Uses this website to match zipcodes with precints: https://www1.nyc.gov/site/nypd/bureaus/patrol/find-your-precinct.page
	Join the rent data with NYPD complaint data.
	"""
	rentDf['precinct'] = rentDf['addr_zip'].map({
		11226:70,
		10013:1,
		10022:17,
		10018:10,
		11216:79,
		10025:24,
		11372:115,
		11102:114,
		10036:18,
		11377:108,
		11211:94,
		10031:30,
		11237:90,
		10032:33,
		11221:83,
		11205:79,
		11223:62,
		10021:19,
		10003:9,
		10023:20,
		11206:90,
		11222:94,
		10019:18,
		10010:13,
		11201:84,
		11106:114,
		10128:19,
		11104:108,
		11213:77,
		11209:68,
		10004:1, 
		11233:81,
		11217:84,
		11231:76,
		11354:109,
		10017:17,
		10014:6,
		10467:52,
		10040:34,
		10069:20,
		10038:1,
		11220:72,
		11103:114,
		10012:1,
		10016:17,
		10463:50,
		10030:32,
		10011:10,
		10001:10,
		11230:66,
		11215:78,
		10027:28,
		10009:9,
		11210:63,
		11238:77,
		10065:19,
		11105:114,
		11375:112,
		11219:66,
		11249:90,
		11379:104,
		10075:19,
		11385:104,
		10024:20,
		10005:1,
		11214:62,
		11232:72,
		11373:110,
		10006:1,
		10280:1,
		10028:19,
		11203:67,
		10039:32,
		10035:25,
		11235:61,
		11218:66,
		11374:112,
		11432:107,
		11225:71,
		11229:61,
		11207:75,
		10026:28,
		10002:7,
		11364:111,
		10044:114,
		11435:103,
		10029:23,
		11101:108,
		10034:34,
		11234:63,
		10033:34,
		11358:109,
		11415:102,
		10462:49,
		11204:62,
		11228:68,
		10456:44,
		11355:109,
		11208:75,
		10471:50,
		11370:114,
		10454:40,
		11212:73,
		10007:1,
		10473:43,
		10458:48,
		11367:107,
		11224:60,
		11109:108,
		10282:1,
		11236:69,
		10452:44,
		10451:44,
		11368:110,
		11421:102,
		11357:109,
		10468:52,
		10301:120,
		11366:107, 
		10037:25,
		10472:43,
		11434:113,
		11378:104, 
		10457:48, 
		11361:111,
		10459:41,
		10455:40,
		10466:47,
		11414:106,
		11692:100,
		11694:100, 
		10464:45,
		11428:105,
		11693:100,
		10453:46,
		11417:106,
		10465:45,
		11369:115,
		11691:101,
		11418:102,
		10461:49,
		11365:111,
		10304:120,
		11416:102,
		11360:109,
		10469:49,
		10302:121,
		10460:42,
		11423:103,
		11356:109,
		10314:121,
		10305:122,
		11413:105,
		10470:47,
		11363:111,
	})
	
	rentDf = rentDf.merge(right=nypdDf, how='left', left_on='precinct', right_on='addr_pct_cd')
	return rentDf
	
def joinRent311(rentDf, incidentDf):
	rentDf = rentDf.merge(right=incidentDf, how='left', on='addr_zip')
	return rentDf

# merges NYPD and 311 service data into rental data and returns updated dataframe
def merge_external_data(rental_df, nypd_df, incident_df):
    rental_df = joinRentNYPD(rental_df, nypd_df)
    rental_df = joinRent311(rental_df, incident_df)
    return rental_df

# fits estimator to training set and predicts on test1.csv features
# prints out MSE of predictions against test1.csv
# also calculates and prints std dev & mean of 5-fold CV  on training set
def getResults(estimator, train_features, train_target, test_features, test_target):
    """
	Fits estimator to training set and predicts on features
	Prints out MSE of predictions aginst 
    """
    
    estimator.fit(train_features, train_target)
    test_pred = estimator.predict(test_features)
    cv_results = cross_val_score(estimator, train_features, train_target, cv=5, scoring='neg_mean_squared_error')
    print('Mean Squared Error: ', mean_squared_error(test_target, test_pred))
    print("CV Results (Std Dev): ", np.std(cv_results))
    print("CV Results (Mean): ", np.mean(cv_results))

#To get the feature importances for linear regression scale the features and then use .coef_
#importances for lr are in magnitude while others are in percentage
#I haven't scaled the features so this code doesn't really tell you much
def getSortedImportances(estimator, features_list):
    temp = {}
    importances = estimator.coef_ if isinstance(estimator, LinearRegression) else estimator.feature_importances_*100
    print('\nFeature Importances')
    for i in range(len(importances)):
        temp[features_list[i]] = importances[i]
    
    sorted_temp = sorted(temp, key=temp.get)
    for feat in sorted_temp:
        print(feat, temp[feat])
        
def testHyperparameter(param_grid, train_features, train_target):
    results = {}
    hp_name = list(param_grid.keys())[0]
    hp_values = list(param_grid.values())[0]
    
    # Create decision tree model and tune hyperparameter using GridSearchCV
    dt = DecisionTreeRegressor()
    dt_cv = GridSearchCV(dt, param_grid, cv=5, return_train_score=True)
    dt_cv.fit(train_features, train_target)
    results['best_param'] = dt_cv.best_params_
    results['best_score'] = dt_cv.best_score_
    
    # Grab mean test/train score of 5 splits
    mean_test = dt_cv.cv_results_['mean_test_score']
    mean_train = dt_cv.cv_results_['mean_train_score']

    # Plot mean test/train AUC score against hyperparameter values
    line1, = plt.plot(hp_values, mean_test, 'b', label='Test AUC')
    line2, = plt.plot(hp_values, mean_train, 'r', label='Train AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC Score')
    plt.xlabel(hp_name)
    plt.show()
    
    print(results)
        
        