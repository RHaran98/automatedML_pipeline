import numpy as np
import pandas as pd
from category_encoders import WOEEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import os
from sklearn.metrics import accuracy_score,  precision_score, recall_score, confusion_matrix, f1_score


class Dataset(object):
    def __init__(self):
        # Declare the attributes
        self.whole_df = None
        self.whole_cols = None
        self.dataset_name = None
        self.drop_cols = None
        self.id_cols = None
        self.data = None
        self.features = None
        self.target = None
        self.target_col = None
        self.n_rows = None
        self.n_cols = None
        self.cat_cols = None
        self.num_cols = None
        self.feature_names = None

    @classmethod
    def from_df(cls, df, target="TARGET", drop_cols=None, id_cols=None):
        logging.info("Creating dataframe from dataframe")
        cls = cls()
        assert isinstance(df, pd.DataFrame)
        cls.whole_df = df
        cls.whole_cols = cls.whole_df.columns
        cls.target = target
        cls.dataset_name = "df"
        # Basic checks
        assert len(cls.whole_cols) == len(set(cls.whole_cols))  # Check for duplicate column names
        if target:
            assert target in cls.whole_cols  # Check if target col present

        cls._populate_attributes(drop_cols, id_cols)

        return cls

    @classmethod
    def from_csv(cls, file_path, target="TARGET", sep="~", drop_cols=None, id_cols=None):
        cls = cls()
        # Read in file
        logging.info("Creating dataset from {}".format(file_path))
        cls.whole_df = pd.read_csv(file_path, sep=sep)
        cls.whole_cols = cls.whole_df.columns
        cls.target = target
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        cls.dataset_name = file_name

        # Basic checks
        assert len(cls.whole_cols) == len(set(cls.whole_cols))  # Check for duplicate column names
        if target:
            assert cls.target in cls.whole_cols  # Check if target col present

        cls._populate_attributes(drop_cols, id_cols)

        return cls

    def _populate_attributes(self, drop_cols, id_cols):
        if drop_cols:
            self.drop_cols = list(set(self.whole_cols).intersection(drop_cols))
        else:
            self.drop_cols = []

        if id_cols:
            self.id_cols = list(set(self.whole_cols).intersection(id_cols))
        else:
            self.id_cols = []

        self.whole_df = self.whole_df.replace([-np.inf, np.inf], np.nan)
        self.data = self.whole_df.drop(columns=self.drop_cols + self.id_cols)

        # Shift target to the last
        if self.target:
            assert self.target in self.data.columns  # Check if target is present in the data

            _cols = list(self.data)
            _cols.insert(len(_cols), _cols.pop(_cols.index(self.target)))
            self.data = self.data.ix[:, _cols]

        self._update_columns()

        # Pipeline state variables
        self.enc = None

    def _update_columns(self):
        self.n_rows = self.data.shape[0]
        self.n_cols = self.data.shape[0]
        self.features = self.data.iloc[:,:-1]
        self.feature_names = self.features.columns
        if self.target:
            self.target_col = self.data[self.target]
        else:
            self.target_col = None
        self.cat_cols = self.features.dtypes[self.features.dtypes == "object"].index
        self.num_cols = self.features.dtypes[~(self.features.dtypes == "object")].index
    #
    # def handle_missing_values(self, threshold=0.3):
    #     thresh = len(self.data) * threshold
    #     self.data.dropna(thresh=thresh, axis=1, inplace=True)
    #
    # def one_hot_encoding(self, cols=None):
    #     if cols:
    #         self.data = pd.get_dummies(self.data)
    #         new_levels = list(set(self.data.columns) - set(cols))
    #         old_levels = list(set(cols) - set(self.data.columns))
    #         # self.data.drop(columns=new_levels,inplace=True)
    #         for i in new_levels:
    #             org_i = i.split('_')[0]
    #             org_i_alt = org_i + "_" + "Other"
    #             if org_i_alt in self.data.columns:
    #                 self.data[org_i_alt] = self.data[org_i_alt] | self.data[i]
    #         self.data.drop(columns=new_levels, inplace=True)
    #
    #         for i in old_levels:
    #             self.data[i] = 0
    #     else:
    #         self.data = pd.get_dummies(self.data)
    #
    # def collapse_levels(self, cols=None, threshold=0.01):
    #     pass
    #
    # def woe_encoding(self):
    #     self.enc = WOEEncoder(cols=self.cat_cols).fit(self.data[self.features],self.data[self.target])
    #     self.data = self.enc.transform(self.data)
    #     self._update_columns()
    #
    # def auto_drop_columns(self, cols=None, levels_threshold=0.3,missing_thresh=0.3):
    #     thresh = levels_threshold*len(self.data)
    #     if cols:
    #         columns = cols
    #     else:
    #         columns = self.cat_cols
    #
    #     for col in columns:
    #         if len(self.data[col].unique()) > thresh:
    #             self.data.drop(columns=col,inplace=True)
    #
    #     self.data.dropna(axis=1, thresh=int(missing_thresh * len(self.data)))
    #
    #     self._update_columns()

    def test_train_split(self, split_ratio=0.3, seed=1234):
        train = self.data.sample(frac=1-split_ratio,random_state=seed)
        test = self.data.drop(train.index)

        d_train = Dataset.from_df(train, target=self.target)
        d_test = Dataset.from_df(test, target=self.target)

        return d_train, d_test


# class Model(object):
#     def __init__(self,model=None):
#         self.model = model
#         self.feature_names = None
#         self.target = None
#         self.isTrained = False
#
#     def fit(self, dataset):
#         self.feature_names = dataset.feature_names
#         self.target = dataset.target
#         self.model.fit(dataset.data[self.feature_names], dataset.data[self.target])
#         self.isTrained = True
#
#     def predict(self, dataset):
#         assert self.isTrained
#         features = np.array(dataset.data[self.feature_names])
#         return self.model.predict(features)
#
#     def load_model(self, model_path):
#         assert os.path.isfile(model_path)
#         self.isTrained = True
#         model_state = joblib.load(model_path)
#         self.model = model_state["model"]
#         self.feature_names = model_state["feature_names"]
#         self.target = model_state["target"]
#
#     def save_model(self,model_path):
#         assert self.isTrained
#         model_state = {"model": self.model, "feature_names": self.feature_names, "target": self.target}
#         joblib.dump(model_state, model_path)

class AutoDropColumns(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, missing_thresh=0.3, levels_thresh=0.3):
        self.missing_thresh = missing_thresh
        self.levels_thresh = levels_thresh

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        missing_thresh = len(X) * self.missing_thresh
        levels_thresh = len(X) * self.levels_thresh
        X.dropna(thresh=missing_thresh, axis=1, inplace=True)
        cols = X.dtypes[X.dtypes == "object"].index
        for i in cols:
            if len(X[i].unique()) > levels_thresh:
                logging.debug("Dropping column {0}".format(i))
                X.drop(columns=i, inplace=True)
        logging.debug("Columns {0}".format(X.columns))
        return X


class ColumnsSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, cols=None):
        if not cols:
            self.cols = []
        else :
            self.cols = cols

    def fit(self, X, y=None):
        self.cols = X.columns
        return self

    def transform(self, X, y=None):
        old_cols = list(set(self.cols) - set(X.columns))
        for i in old_cols:
            X[i] = 0

        return X[self.cols]


class AutomatedPipeline:
    def __init__(self):
        self.pipeline = []

    def add_woe_encoding(self):
        self.pipeline.append(("WOEncoder",WOEEncoder()))

    def add_one_hot_encoding(self):
        self.pipeline.append(("WOEncoder", OneHotEncoder(handle_unknown="ignore")))

    def add_column_filter(self,suffix=""):
        self.pipeline.append(("ColFilter"+str(suffix), ColumnsSelector()))

    def add_column_auto_drop(self):
        self.pipeline.append(("ColDrop", AutoDropColumns()))

    @classmethod
    def make_pipeline(cls, clf):
        cls = cls()
        cls._build_pipeline()
        cls.pipeline.append(("Classifier",clf))
        cls.pipeline = Pipeline(cls.pipeline)
        return cls

    @classmethod
    def load_pipeline(cls, pipeline_state):
        cls = cls()
        cls.pipeline = pipeline_state["pipeline"]
        return cls

    def _build_pipeline(self):
        self.add_column_auto_drop()
        self.add_column_filter(0)
        self.add_woe_encoding()
        self.add_column_filter(1)

    def save_pipeline(self,save_path,**kwargs):
        pipeline_state = {"pipeline": self.pipeline, **kwargs}
        joblib.dump(pipeline_state, save_path)


class Results(object):
    def __init__(self, y_true, y_pred, bins=None):
        self.y_true = self.standardize(y_true)
        self.y_pred = self.standardize(y_pred)
        self.y_pred_labels = np.round(self.y_pred)
        self.results_df = pd.DataFrame({"Bad":y_true,"Score":y_pred})

        self.gini_table = None
        self.bins = bins
        self.gini_val = None
        self._make_gini_table()

        self.metrics_dict = {}
        self.metrics_table = None
        self._calculate_common_metrics()

    @staticmethod
    def standardize(x):
        return list(np.array(x).ravel())

    def _make_gini_table(self, n_bins=10):
        self.results_df["Good"] = 1 - self.results_df["Bad"]
        if self.bins is not None:
            self.results_df["Bin"] = pd.cut(self.results_df["Score"],bins=self.bins)
        else:
            self.results_df["Bin"], self.bins = pd.cut(self.results_df["Score"],n_bins,retbins=True)
        grouped = self.results_df.groupby("Bin",as_index=False)
        lower = grouped.min().Score
        upper = grouped.max().Score
        self.gini_table = pd.DataFrame({"Lower":lower, "Upper":upper, "Count":grouped.sum().Good + grouped.sum().Bad, "Bad":grouped.sum().Bad, "Good":grouped.sum().Good})
        self.gini_table = (self.gini_table.sort_values(by='Lower')).reset_index(drop=True)

        self.gini_table["Bad rate"] = (self.gini_table["Bad"] / self.gini_table["Count"])*100
        self.gini_table["Bad Capture rate"] = np.cumsum(self.gini_table["Bad"]/self.gini_table["Bad"].sum())*100
        self.gini_table["Good rate"] = (self.gini_table["Bad"] / self.gini_table["Count"])*100
        self.gini_table["Good Capture rate"] = np.cumsum(self.gini_table["Good"]/self.gini_table["Good"].sum())*100

        self.gini_table['KS'] = np.abs(self.gini_table["Good Capture rate"] - self.gini_table["Bad Capture rate"])
        self.gini_table["Gini"] = (self.gini_table["Bad Capture rate"] + self.gini_table["Bad Capture rate"].shift(1).fillna(0)) * (self.gini_table["Good Capture rate"] - self.gini_table["Good Capture rate"].shift(1).fillna(0)) * (0.5/100.0)
        self.gini_val = np.abs(0.5 - (self.gini_table["Gini"].sum()/100.0))

    def _calculate_common_metrics(self):
        self.metrics_dict["Accuracy"] = accuracy_score(self.y_true, self.y_pred_labels)
        self.metrics_dict["F1"] = f1_score(self.y_true, self.y_pred_labels)
        self.metrics_dict["Precision"] = precision_score(self.y_true, self.y_pred_labels)
        self.metrics_dict["Recall"] = recall_score(self.y_true, self.y_pred_labels)
        self.metrics_dict["GINI"] = self.gini_val
        con_matrix = confusion_matrix(self.y_true, self.y_pred_labels)
        tn, fp, fn, tp = con_matrix.ravel()
        self.metrics_dict["True negatives"] = tn
        self.metrics_dict["False positives"] = fp
        self.metrics_dict["False negatives"] = fn
        self.metrics_dict["True positives"] = tp
        self.metrics_table = pd.DataFrame.from_dict(self.metrics_dict, orient="index")
