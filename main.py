from LoadData import LoadData
import configurations.configurations as conf
import argparse




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-path_configuration", type=str, default="./configurations/default.json", help="specify the path to the configuration file"
    )

    args = parser.parse_args()

    path_configuration = args.path_configuration
    configuration = conf.get_configuration(path_configuration)


    conf_data = configuration["data"]
    load_data = LoadData(base_path=conf_data["base_path"], exact_polytopes=conf_data["exact_polytopes"])
    load_data.add_dataset(conf_data["train-test-data"][0], train_data=True)
    #load_data.add_dataset(conf_data["train-test-data"][1], train_data=False)
    
    load_data.get_data_split(test_split_size=conf_data["train-test-split"], dev_split_size=conf_data["train-eval-split"])