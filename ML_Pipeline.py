from automation_utils import Dataset, Results, AutomatedPipeline
import argparse
from argparse import Namespace
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
import os
import logging
import joblib
import pandas as pd
from openpyxl import load_workbook


def get_arguments():
    parser = argparse.ArgumentParser(description="Automated ML pipeline")
    subparser = parser.add_subparsers(help="Mode to run application in")

    parser.add_argument("-v", "--verbosity", type=int, metavar="x", dest="verbosity",
                            help="Set the verbosity level of the app\n\t0 : No text\n\t1 : Print steps\n\t2 : Print "
                            "data",
                            required=False, default=0, choices=[0, 1, 2])


    train_parser = subparser.add_parser("train")
    train_parser.set_defaults(mode="train")
    oot_parser = subparser.add_parser("oot")
    oot_parser.set_defaults(mode="oot")
    test_parser = subparser.add_parser("test")
    test_parser.set_defaults(mode="test")

    train_parser.add_argument("--dataset", metavar="*.csv", dest="dataset_path", type=str, help="Path to the dataset", required=True)
    train_parser.add_argument("--oot_dataset", metavar="*.csv", dest="oot_dataset_path", type=str, help="Path to the oot dataset",
                              required=False,default=None)
    train_parser.add_argument("--model_path", metavar="*.sav", type=str, help="Path to the model dump", required=False,default=None)
    train_parser.add_argument("--test_dataset", metavar="*.csv", dest="test_dataset_path", type=str, help="Path to the test dataset",
                              required=False,default=None)
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

    oot_parser.add_argument("--oot_dataset", metavar="*.csv", dest="oot_dataset_path",type=str, help="Path to the oot dataset", required=True)
    oot_parser.add_argument("--model_path", metavar="*.sav", type=str, help="Path to the model dump", required=True)
    oot_parser.add_argument("--target", metavar="col", type=str, help="Name of the target columns", required=False,default="TARGET")
    oot_parser.add_argument("--sep", "--seperator", type=str, metavar="c", dest="seperator_character",
                            help="Character to delimit columns in files", required=False, default="~")

    test_parser.add_argument("--test_dataset", metavar="*.csv", dest="test_dataset_path",type=str, help="Path to the test dataset", required=True)
    test_parser.add_argument("--model_path", metavar="*.sav", type=str, help="Path to the model dump", required=True)
    test_parser.add_argument("--sep", "--seperator", type=str, metavar="c", dest="seperator_character",
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
    directory = os.path.dirname(dataset_path)

    dataset = Dataset.from_csv(dataset_path, target=target, drop_cols=drop_columns, id_cols=id_columns, sep=sep)
    train_dataset, test_dataset = dataset.test_train_split()
    automated_pipeline = AutomatedPipeline.make_pipeline(clf)
    automated_pipeline.pipeline.fit(train_dataset.features, train_dataset.target_col)

    y_pred_train = automated_pipeline.pipeline.predict_proba(train_dataset.features)[:,1]
    y_true_train = train_dataset.target_col
    train_results = Results(y_true=y_true_train, y_pred=y_pred_train)

    y_pred = automated_pipeline.pipeline.predict_proba(test_dataset.features)[:,1]
    y_true = test_dataset.target_col
    test_results = Results(y_true=y_true, y_pred=y_pred, bins=train_results.bins)

    feature_importance = automated_pipeline.pipeline.named_steps["Classifier"].feature_importances_
    features = automated_pipeline.pipeline.named_steps['ColFilter1'].cols

    var_imp = pd.DataFrame({"Feature":features, "Importance": feature_importance})
    var_imp.sort_values(["Importance"], ascending=False, inplace=True)

    results_path = os.path.join(directory, dataset.dataset_name + "_results.xlsx")
    with pd.ExcelWriter(results_path) as writer:
        var_imp.to_excel(writer, sheet_name="VarImp")
        train_results.gini_table.to_excel(writer, sheet_name="Train_GINI")
        train_results.metrics_table.to_excel(writer, sheet_name="Train_metrics", header=None)
        test_results.gini_table.to_excel(writer, sheet_name="Test_GINI")
        test_results.metrics_table.to_excel(writer, sheet_name="Test_metrics", header=None)
    pipeline_save_path = os.path.join(directory,"pipeline.sav")
    automated_pipeline.save_pipeline(pipeline_save_path, bins=train_results.bins, target=target)

    if args.oot_dataset_path:
        args.model_path = pipeline_save_path
        oot(args)

    if args.test_dataset_path:
        args.model_path = pipeline_save_path
        test(args)


def oot(args):
    dataset_path = args.oot_dataset_path
    model_path = args.model_path
    target = args.target
    sep = args.seperator_character

    directory = os.path.dirname(dataset_path)

    pipeline_state = joblib.load(model_path)
    oot_dataset = Dataset.from_csv(dataset_path, target=pipeline_state["target"], sep=sep)
    automated_pipeline = AutomatedPipeline.load_pipeline(pipeline_state)


    y_pred_oot = automated_pipeline.pipeline.predict_proba(oot_dataset.features)[:,1]
    y_true_oot = oot_dataset.target_col

    results_path = os.path.join(directory, oot_dataset.dataset_name + "_results.xlsx")
    oot_results = Results(y_true=y_true_oot, y_pred=y_pred_oot, bins=pipeline_state["bins"])
    with pd.ExcelWriter(results_path) as writer:
        if os.path.isfile(results_path):
            writer.book = load_workbook(results_path)
        oot_results.gini_table.to_excel(writer, sheet_name="OOT_GINI")
        oot_results.metrics_table.to_excel(writer, sheet_name="OOT_metrics", header=None)

def test(args):
    dataset_path = args.test_dataset_path
    model_path = args.model_path
    sep = args.seperator_character

    directory = os.path.dirname(dataset_path)
    pipeline_state = joblib.load(model_path)
    test_dataset = Dataset.from_csv(dataset_path, target=None, sep=sep)
    automated_pipeline = AutomatedPipeline.load_pipeline(pipeline_state)
    bins = pipeline_state["bins"]

    export_file_path = os.path.join(directory, test_dataset.dataset_name + "_predictions" + ".txt")
    export_df = test_dataset.whole_df.copy()
    export_df["Probability"] = automated_pipeline.pipeline.predict_proba(test_dataset.features)[:,1]
    export_df["Bands"] = pd.cut(export_df["Probability"],bins=bins)
    export_df.to_csv(export_file_path,index=None)

def main():
    args = get_arguments()
    mode = args.mode
    verbosity = args.verbosity
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbosity == 2:
        logging.basicConfig(level=logging.DEBUG)

    if mode == "train":
        train(args)
    elif mode == "test":
        test(args)
    elif mode == "oot":
        oot(args)


if __name__ == "__main__":
    main()
