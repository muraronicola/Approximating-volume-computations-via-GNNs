# Approximating volume computations via GNNs
Advanced Topics in Machine Learning and Optimization

---

## Creating the Environment

To create the environment, run the following command:

```bash
conda env create -f environment.yml
```

To execute the other steps, is first necessary to activate the environment:

```bash
conda activate ADVML
```
---




## Generating the data
There are already some pre-generated data in the `data` folder. However, if you want to generate new data, you can run the following command:

```bash
python3 ./GenerateData.py <dimensions> <constraints> <number_polytopes> <uniform_distribution> <max_volume> <seed> <only_exact>
```
**Arguments:**
- `<dimensions>`: Number of dimensions of the polytopes.
- `<constraints>`: Number of constraints of the polytopes.
- `<number_polytopes>`: Number of polytopes to generate.
- `<uniform_distribution>`: If set to `True`, the polytopes will be generated using a uniform distribution. If set to `False`, the polytopes will be generated using a normal distribution.
- `<max_volume>`: Maximum volume of the polytopes.
- `<seed>`: Seed for the random number generator.
- `<only_exact>`: If set to `True`, only polytopes without reduntant constraints will be generated. If set to `False`, polytopes with redundant constraints will also be generated.



---

## Training the GNN


For executing the code, run the following command:

```bash
python3 ./main.py <path_configuration>
```

**Arguments:**

- `<path_configuration>`: Path of the file containing the configuration.

---

## Changing the configuration

The default configuration file is located in the `configurations` folder.
The file contains some important parameters such as the data used for training, the number of epochs, the batch size, the learning rate and the output directory for the results.

---

## Dummy Classifier

The file `dummy_classifier.py` contains the implementation of a dummy classifier that can be used to compare the results of the GNN with a simple baseline. For executing the dummy classifier, run the following command:

```bash
python3 ./DummyClassifier.py <path_configuration>
```
**Arguments:**
- `<path_configuration>`: Path of the file containing the configuration.


## Evaluating the results
To evaluate the results of the GNN, you can view the jupyter notebook in the `analyze_results` folder. The notebook contains the code to load the results and visualize them.