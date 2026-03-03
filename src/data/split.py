import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # read raw data
    df = pd.read_csv( 'data/raw_data/raw.csv' )
    
    # cut the columns of date and destination
    X = df.iloc[ : , 1 : -1 ]
    # slice of the destination
    y = df.iloc[ : , -1 ]

    # split train and test with 80% and 20% of dataset with reproduceable random
    X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.2 , random_state = 43 )

    # put the data to processed folder
    X_train.to_csv( "data/processed_data/X_train.csv" , index = False)
    X_test.to_csv( "data/processed_data/X_test.csv" , index = False)
    y_train.to_csv( "data/processed_data/y_train.csv" , index = False)
    y_test.to_csv( "data/processed_data/y_test.csv" , index = False)

if __name__ == "__main__":
    main()
