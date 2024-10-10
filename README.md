# Running Python Code 


## 1. Unzip the file
The folder consists of following structure
```
├── configs
│   ├── config_1.txt
│   ├── config_2.txt
│   └── config-readme.txt
├── genetic_algo.py
├── main.py
├── notebooks
│   ├── packing_animation.mp4
│   └── TheFancyStuff.ipynb
├── output
│   ├── Cust_Eval.csv
│   ├── Eval.csv
│   ├── selection_roulette_28_generation_config_1.csv
│   ├── selection_roulette_crossover_mutation_0.05_28_generation_config_1.csv
│   ├── selection_tournament_17_generation_config_1.csv
│   ├── selection_tournament_28_generation_config_1.csv
│   ├── selection_tournament_crossover_mutation_0.05_28_generation_config_1.csv
│   ├── selection_tournament_crossover_mutation_0.05_29_generation_config_1.csv
│   └── selection_tournament_crossover_mutation_0.05_custom_fitness_29_generation_config_1.csv
├── __pycache__
│   └── genetic_algo.cpython-312.pyc
└── README.md

    
```    
Unzip the zip file
```
unzip ps2-gopi.zip
```

## 2. Install all the dependencies
Ensure you are within `ps2-gopi` folder.
```
pip install -r requirements.txt
```

This may take a minute or two.

## 3. Run the Tree

### 3.1 Change Directory to Genetic Algo
```
cd GeneticAlgo
```

### 3.2 Run the Code
```
python3 main.py --config configs/config_1.txt --save_output --enable_crossover --enable_mutation --selection 1
```

List of Available Params
```
--config # path of the config file
--save_output # to save the output in output folder
--enable_crossover # for enabling crossover feature
--enable_mutation # for enabling mutation feature
--selection # selection criteria, tournament (0), roulette (1)
--enable_prune # to enable or disable pruning
--early_stop # to enable early stopping
--evaluate_population_sizes to run in eval mode
--num_trials trial size for eval mode 

```

## 4. Question References

Most of the written summary are present in the report. The problem based ones are in 3 places, 
1. code for Genetic Knapsack in the GeneticAlgo Folder
2. notebooks folder contains notebook that takes the data saved in output folder to visualize them
3. Knapsack Problem solved using dynamic programming is present in the above notebook in the end







