import argparse
import os
from module.utils import dic_functions
from learner import trainer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()

parser.add_argument("--run_type", default="ours", help="Run Type")
parser.add_argument("--dataset_in",default="CMNIST", help="Name of the Dataset")
parser.add_argument("--model_in_d", default="MLP", help="Name of the model")
parser.add_argument("--model_in_b", default="MLP", help="Name of the model")
parser.add_argument("--train_samples", default=3000, type=int,help="Number of training samples")
parser.add_argument("--bias_ratio", default=0.02, type = float,help="Bias ratio")
parser.add_argument("--bias_ratio_og", default=0.05, type = float,help="Bias ratio")
parser.add_argument("--runs", default=3, type = int,help="Number of runs")
parser.add_argument("--reduce", default = 1, type = int, help = "Reduce the number of samples")
parser.add_argument("--type", default = 1, type = int, help = "CIFAR type")
parser.add_argument("--severity", default = 4, type = int, help = "severity")
parser.add_argument("--epoch_preprocess", default=100, type = int, help = "Number of epochs for preprocessing")
parser.add_argument("--preprocess", default = 'none', help = "Preprocessing")
parser.add_argument("--seed_init", default = 1, type = int, help = "Seed for the initialisation")
parser.add_argument("--mix_up_val", default = 0.9, type = float, help = "Mix up value")
parser.add_argument("--thresh", default=0.95, help="Threshold for the MixUP")
parser.add_argument("--loss_contr", default=1.0, help="Loss contribution", type = float)
args = parser.parse_args()


write_to_file = dic_functions['write_to_file']

run = trainer(args)

for run_num in range(args.runs):
    run.get_results(run_num + args.seed_init)

write_to_file('results_text/'+ run.name_run+ '.txt', '---------------------\n')