import json
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def main():
    X_test = pd.read_csv( "data/processed_data/X_test_scaled.csv" )
    y_test = pd.read_csv( "data/processed_data/y_test.csv" ).squeeze( "columns" )

    model = joblib.load( "models/gbr_model.pkl" )
    y_pred = model.predict( X_test )

    # prediction.csv
    pd.DataFrame( { "y_true": y_test , "y_pred": y_pred } ).to_csv( "data/prediction.csv" , index = False )

    # metrics
    scores = {
        "mse": float( mean_squared_error( y_test , y_pred ) )
        ,"r2": float( r2_score( y_test , y_pred ) )
    }
    with open( "metrics/scores.json" , "w" ) as f:
        json.dump( scores , f , indent = 2 )

if __name__ == "__main__":
    main()
