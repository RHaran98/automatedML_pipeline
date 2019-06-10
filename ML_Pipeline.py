from automation_utils import Dataset, Results, AutomatedPipeline
import argparse
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
import os
import logging
import joblib
logging.basicConfig(level=logging.DEBUG)

def get_arguments():
    parser = argparse.ArgumentParser(description="Automated ML pipeline")
    subparser = parser.add_subparsers(help="Mode to run application in")
    train_parser = subparser.add_parser("train")
    train_parser.set_defaults(mode="train")
    oot_parser = subparser.add_parser("oot")
    oot_parser.set_defaults(mode="oot")

    train_parser.add_argument("--dataset", metavar="*.csv", dest="dataset_path", type=str, help="Path to the dataset", required=True)
    train_parser.add_argument("--target", metavar="col", type=str, help="Name of the target columns", required=False,default="TARGET")
    train_parser.add_argument("--clf", metavar="clf", type=str, help="Classifier to use", required=False, default="xgb",
                              choices=["rf", "dt", "gb", "xgb"])
    train_parser.add_argument("--drop", "--drop_columns", metavar="C", dest="drop_columns", type=str, nargs="*",
                              help='Column names to drop', required=False, default=[])
    train_parser.add_argument("--id", "--id_columns", metavar="I", dest="id_columns", type=str, nargs='*',
                              help="ID columns", required=False, default=[])
    train_parser.add_argument("--sep", "--seperator", type=str, metavar="c", dest="seperator_character",
                              help="Character to delimit columns in files", required=False, default="~")
    train_parser.add_argument("-v", "--verbosity", type=int, metavar="x", dest="verbosity",
                              help="Set the verbosity level of the app\n\t0 : No text\n\t1 : Print steps\n\t2 : Print "
                                   "data",
                              required=False, default=0, choices=[0, 1, 2])

    oot_parser.add_argument("--dataset", metavar="*.csv", dest="dataset_path",type=str, help="Path to the dataset", required=True)
    oot_parser.add_argument("--model_path", metavar="*.sav", type=str, help="Path to the model dump", required=True)
    oot_parser.add_argument("--target", metavar="col", type=str, help="Name of the target columns", required=False,default="TARGET")
    oot_parser.add_argument("--sep", "--seperator", type=str, metavar="c", dest="seperator_character",
                              help="Character to delimit columns in files", required=False, default="~")
    args = parser.parse_args()

    return args


def train(args):
    dataset_path = args.dataset_path
    target = args.target
    drop_columns = args.drop_columns
    id_columns = args.id_columns
    sep = args.seperator_character
    clf = args.clf
    clf = {"xgb": XGBClassifier(), "rf": RandomForestClassifier(), "dt": DecisionTreeClassifier(),
           "gb": GradientBoostingClassifier()}[clf]

    dataset = Dataset.from_csv(dataset_path, target=target, drop_cols=drop_columns, id_cols=id_columns, sep=sep)
    train_dataset, test_dataset = dataset.test_train_split()
    automated_pipeline = AutomatedPipeline.make_pipeline(clf)
    automated_pipeline.pipeline.fit(train_dataset.features, train_dataset.target_col)
    automated_pipeline.save_pipeline(os.path.join(os.path.dirname(dataset_path),"pipeline.sav"))

    y_pred = automated_pipeline.pipeline.predict_proba(test_dataset.features)[:,1]
    y_true = test_dataset.target_col

    logging.debug("y_pred {}".format(y_pred))
    results = Results(y_true=y_true, y_pred=y_pred)
    results.gini_table.to_csv(os.path.join(os.path.dirname(dataset_path), "Test_Summary.csv"))


def oot(args):
    dataset_path = args.dataset_path
    model_path = args.model_path
    target = args.target
    sep = args.seperator_character
    # model_state =  Model.load_model()
    oot_dataset = Dataset.from_csv(dataset_path, target=target, sep=sep)
    model_state = joblib.load(model_path)
    automated_pipeline = AutomatedPipeline.load_pipeline(model_path)
    y_pred_oot = automated_pipeline.pipeline.predict_proba(oot_dataset.features)[:,1]
    y_true_oot = oot_dataset.target_col
    oot_results = Results(y_true=y_true_oot, y_pred=y_pred_oot)
    oot_results.gini_table.to_csv(os.path.join(os.path.dirname(dataset_path), "OOT_Summary.csv"))


def main():
    args = get_arguments()
    mode = args.mode
    if mode == "train":
        train(args)
    elif mode == "oot":
        oot(args)


if __name__ == "__main__":
    main()
