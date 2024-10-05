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

    args = parser.parse_args()

    test = KnapSack(args.config, args.selection, args.enable_crossover, 
            args.enable_mutation, args.selection_rate, 
            args.mutation_rate, args.save_output, 
            args.early_stop, args.tournament_size)

    test.run()
