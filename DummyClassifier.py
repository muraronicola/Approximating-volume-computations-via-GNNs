import configurations.configurations as conf
import argparse
import numpy as np

from src.utils import mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split


# Main function to run the dummy classifier evaluation
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_configuration", type=str, default="./configurations/default.json", help="specify the path to the configuration file")
    
    args = parser.parse_args()
    path_configuration = args.path_configuration
    
    configuration = conf.get_configuration(path_configuration)
    
    conf_data = configuration["data"]
    
    
    X = np.load("./data/" + conf_data["data"][0] + "/exact_politopes_x.npy", allow_pickle=True)
    y = np.load("./data/" + conf_data["data"][0] + "/exact_politopes_y.npy", allow_pickle=True)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    classifier_types = ["mean", "median", "quantile", "constant"]

    for type in classifier_types:

        if type == "mean":
            dummy_clf = DummyRegressor(strategy="mean")
        elif type == "median":
            dummy_clf = DummyRegressor(strategy="median")
        elif type == "quantile":
            dummy_clf = DummyRegressor(strategy="quantile", quantile=0.5)
        elif type == "constant":
            dummy_clf = DummyRegressor(strategy="constant", constant=250)
        
        dummy_clf.fit(X_train, y_train)
        pred_test = dummy_clf.predict(X_test)
        
        mae = mean_absolute_error(pred_test, y_test)
        print(f"MAE of Dummy Classifier ({type}): {mae}\n")
        


if __name__ == "__main__":
    main()