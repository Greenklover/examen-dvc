import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def main():
    X_train = pd.read_csv( "data/processed_data/X_train_scaled.csv" )
    y_train = pd.read_csv( "data/processed_data/y_train.csv" ).squeeze( "columns" )

    model = GradientBoostingRegressor( random_state = 43 )
    grid = {
        "n_estimators": [ 100 , 200 ]
        ,"learning_rate": [ 0.05 , 0.1 ]
        ,"max_depth": [ 2 , 3 ]
    }

    gs = GridSearchCV(
        estimator = model
        ,param_grid = grid
        ,scoring = "r2"
        ,cv = 5
        ,n_jobs = -1
    )
    gs.fit( X_train, y_train )

    joblib.dump( gs.best_params_ , "models/best_params.pkl" )

if __name__ == "__main__":
    main()
