import argparse
def get_arguments():
    parser = argparse.ArgumentParser(description='ML for dropout prediction')
	parser.add_argument("--train", type=bool, default=1,help="set a module in training mode")
	parser.add_argument("--test", type=bool, default=0,help="set a module in testing mode")
	parser.add_argument("--cv", type=int, default=3,help="KFolds")
	args = parser.parse_args()
    return parser.parse_args()



"""
parser = argparse.ArgumentParser(description='ML for dropout prediction')
parser.add_argument("--train", type=bool, default=1,help="set a module in training mode")
parser.add_argument("--test", type=bool, default=0,help="set a module in testing mode")
parser.add_argument("--cv", type=int, default=3,help="KFolds")
"""
args = get_arguments()
print(args.cv)