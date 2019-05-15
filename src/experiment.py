from learner import learn, predict
import argparse
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='ML for dropout prediction')
parser.add_argument("--train", type=bool, default=1,help="set a module in training mode")
parser.add_argument("--test", type=bool, default=0,help="set a module in testing mode")
parser.add_argument("--cv", type=int, default=3,help="KFolds")
parser.add_argument("--smote", type=int, default=None,help="whether to use smote")
args = parser.parse_args()

if args.train:
    file_name ="../data/train_data_week_1_challenge.csv"
    learn(file_name,  args.cv, args.smote)
if args.test:
    file_name ="../data/test_data_week_1_challenge.csv"
    label ="../data/test_label_week_1_challenge.csv"
    predict(file_name, label, args.smote)
