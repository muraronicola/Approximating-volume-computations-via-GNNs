# Approximating Volume Computations via GNNs

Project for Advanced Topics in Machine Learning and Optimization

---

## Creating the Environment

To create the environment, run the following command:

```bash
conda env create -f environment.yml
```

Before executing any other steps, activate the environment:

```bash
conda activate ADVML
```

---

## Generating the Data

Some pre-generated data is available in the `data` folder. To generate new data, run:

```bash
python3 ./GenerateData.py <dimensions> <constraints> <number_polytopes> <uniform_distribution> <max_volume> <seed> <only_exact>
```

**Arguments:**
- `<dimensions>`: Number of dimensions of the polytopes.
- `<constraints>`: Number of constraints for the polytopes.
- `<number_polytopes>`: Number of polytopes to generate.
- `<uniform_distribution>`: If `True`, polytopes are generated using a uniform distribution; if `False`, a normal distribution is used.
- `<max_volume>`: Maximum volume of the polytopes.
- `<seed>`: Random seed.
- `<only_exact>`: If `True`, only polytopes without redundant constraints are generated; if `False`, polytopes may include redundant constraints.

---

## Training the GNN

To train the GNN, run:

```bash
python3 ./main.py <path_configuration>
```

**Arguments:**
- `<path_configuration>`: Path to the configuration file.

---

## Dummy Classifier

The file `dummy_classifier.py` implements a baseline dummy classifier to compare against the GNN results. To execute it, run:

```bash
python3 ./DummyClassifier.py <path_configuration>
```

**Arguments:**
- `<path_configuration>`: Path to the configuration file.

---

## Modifying the Configuration

The default configuration files are located in the `configurations` folder.

These files include parameters such as:
- Training dataset
- Number of epochs
- Batch size
- Learning rate
- Output directory for results

To change the dataset used for training, modify the `data` and `data-split` fields in the configuration file:

- `data`: List of data file names.
- `data-split`: Corresponding list indicating how each dataset is used (`tr` for training, `de` for validation, `te` for testing). Multiple options are available, separated by `-` (e.g., `tr-de-te` or `tr-de`).

Both fields are arrays and each entry in `data` must correspond to an entry in `data-split`.

---

## Evaluating the Results

To evaluate the GNN's results, use the Jupyter notebook in the `analyze_results` folder. The notebook contains tools to load and visualize the results.
