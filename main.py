import argparse
from genetic_algo import KnapSack

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default="./configs/config_1.txt")
    parser.add_argument('--selection', type=int, default=0)
    parser.add_argument('--enable_crossover', default=False, action="store_true")
    parser.add_argument('--enable_mutation', default=False, action='store_true')
    parser.add_argument('--selection_rate', type=float, default=1)
    parser.add_argument('--tournament_size', type=int, default=2)
    parser.add_argument('--mutation_rate', type=float, default=0.05)
    parser.add_argument('--save_output', default=False, action='store_true')
    parser.add_argument('--early_stop', default=False, action='store_true')
    parser.add_argument('--evaluate_population_sizes', default='False', action='store_true')
    parser.add_argument('--num_trials', type=int, default=30)
    parser.add_argument('--custom_fit', default=False, action='store_true')

    args = parser.parse_args()

    test = KnapSack(args.config, args.selection, args.enable_crossover, 
            args.enable_mutation, args.selection_rate, 
            args.mutation_rate, args.save_output, 
            args.early_stop, args.custom_fit,
            args.tournament_size)

    if args.evaluate_population_sizes is True:
        # Run in Evaluation mode
        print("Running in Eval Mode:")
        population_sizes = [i for i in range(10, 40, 3)][:10]

        for population_size in population_sizes:
            test.evaluate(population_size, args.num_trials)

    else:
        # Just Run
        print("Running in Normal Mode:")
        test.run()
