import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def main():
    X_train = pd.read_csv( "data/processed_data/X_train_scaled.csv" )
    y_train = pd.read_csv( "data/processed_data/y_train.csv" ).squeeze( "columns" )

    best_params = joblib.load( "models/best_params.pkl" )

    model = GradientBoostingRegressor( random_state = 43 , **best_params )
    model.fit( X_train , y_train )

    joblib.dump( model , "models/gbr_model.pkl" )

if __name__ == "__main__":
    main()
