# %%
# inport packages
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna

# %%
# import data
df = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
# %%
# Assume names and IDs do not affect the results and remove them
df.drop(["Name", "PassengerId"], axis=1, inplace=True)
df_test.drop(["Name", "PassengerId"], axis=1, inplace=True)
# %%
# Handling missing values

# Fill in the missing values in the CryoSleep column with the most common value
df["CryoSleep"].fillna(df["CryoSleep"].mode()[0], inplace=True)
df_test["CryoSleep"].fillna(df_test["CryoSleep"].mode()[0], inplace=True)

# Fill in the missing values in the Cabin column with the most common value
df["Cabin"].fillna(df["Cabin"].mode()[0], inplace=True)
df_test["Cabin"].fillna(df_test["Cabin"].mode()[0], inplace=True)

# Fill in the missing values in the Destination column with the median value
df["Destination"].fillna(df["Destination"].mode()[0], inplace=True)
df_test["Destination"].fillna(df_test["Destination"].mode()[0], inplace=True)

# Fill in the missing values in the Age column with the median value
df["Age"].fillna(df["Age"].median(), inplace=True)
df_test["Age"].fillna(df_test["Age"].median(), inplace=True)

# Fill in the missing values in the HomePlanet column with the most common value
df["HomePlanet"].fillna(df["HomePlanet"].mode()[0], inplace=True)
df_test["HomePlanet"].fillna(df_test["HomePlanet"].mode()[0], inplace=True)

# Fill in the missing values in the VIP column with the most common value
df["VIP"].fillna(df["VIP"].mode()[0], inplace=True)
df_test["VIP"].fillna(df_test["VIP"].mode()[0], inplace=True)

# Fill in the missing values in the RoomService column with the most common value
df["RoomService"].fillna(df["RoomService"].mode()[0], inplace=True)
df_test["RoomService"].fillna(df_test["RoomService"].mode()[0], inplace=True)

# Fill in the missing values in the FoodCourt column with the most common value
df["FoodCourt"].fillna(df["FoodCourt"].mode()[0], inplace=True)
df_test["FoodCourt"].fillna(df_test["FoodCourt"].mode()[0], inplace=True)

# Fill in the missing values in the ShoppingMall column with the most common value
df["ShoppingMall"].fillna(df["ShoppingMall"].mode()[0], inplace=True)
df_test["ShoppingMall"].fillna(df_test["ShoppingMall"].mode()[0], inplace=True)

# Fill in the missing values in the Spa column with the most common value
df["Spa"].fillna(df["Spa"].mode()[0], inplace=True)
df_test["Spa"].fillna(df_test["Spa"].mode()[0], inplace=True)

# Fill in the missing values in the VRDeck column with the most common value
df["VRDeck"].fillna(df["VRDeck"].mode()[0], inplace=True)
df_test["VRDeck"].fillna(df_test["VRDeck"].mode()[0], inplace=True)
# %%
# Split cabin by '/'
df["deck"] = df["Cabin"].str.split("/", expand=True)[0]
df["num"] = df["Cabin"].str.split("/", expand=True)[1].astype("float")
df["side"] = df["Cabin"].str.split("/", expand=True)[2]
df_test["deck"] = df_test["Cabin"].str.split("/", expand=True)[0]
df_test["num"] = df_test["Cabin"].str.split("/", expand=True)[1].astype("float")
df_test["side"] = df_test["Cabin"].str.split("/", expand=True)[2]

df.drop(["Cabin"], axis=1, inplace=True)
df_test.drop(["Cabin"], axis=1, inplace=True)
# %%
df.info()
# %%
# Convert categorical data to numerical dataa
# one-hot encoding
df = pd.get_dummies(
    df,
    columns=["HomePlanet", "CryoSleep", "VIP", "Destination", "deck", "side"],
    sparse=True,
)
df_test = pd.get_dummies(
    df_test,
    columns=["HomePlanet", "CryoSleep", "VIP", "Destination", "deck", "side"],
    sparse=True,
)
# %%
train_feature = df.drop("Transported", axis=1)
test_feature = df_test
train_tagert = df["Transported"]
# Split train data
X_train, X_test, y_train, y_test = train_test_split(
    train_feature, train_tagert, test_size=0.2, random_state=0, stratify=train_tagert
)
# %%
rfc = RandomForestClassifier(random_state=0)
rfc.fit(X_train, y_train)
print("=" * 20)
print("RandomForestClassifier")
print(f"accuracy of train set: {rfc.score(X_train, y_train)}")
print(f"accuracy of test set: {rfc.score(X_test, y_test)}")

xgb = XGBClassifier(random_state=0)
xgb.fit(X_train, y_train)
print("=" * 20)
print("XGBClassifier")
print(f"accuracy of train set: {xgb.score(X_train, y_train)}")
print(f"accuracy of train set: {xgb.score(X_test, y_test)}")

lgb = LGBMClassifier(random_state=0)
lgb.fit(X_train, y_train)
print("=" * 20)
print("LGBMClassifier")
print(f"accuracy of train set: {lgb.score(X_train, y_train)}")
print(f"accuracy of train set: {lgb.score(X_test, y_test)}")

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
print("=" * 20)
print("LogisticRegression")
print(f"accuracy of train set: {lr.score(X_train, y_train)}")
print(f"accuracy of train set: {lr.score(X_test, y_test)}")

svc = SVC(random_state=0)
svc.fit(X_train, y_train)
print("=" * 20)
print("SVC")
print(f"accuracy of train set: {svc.score(X_train, y_train)}")
print(f"accuracy of train set: {svc.score(X_test, y_test)}")
# %%
# Rundom forest with optuna
cv = 5


def objective(trial):

    param_grid_rfc = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 7, 15),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "max_features": trial.suggest_int("max_features", 3, 10),
        "random_state": 0,
    }

    model = RandomForestClassifier(**param_grid_rfc)

    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
rfc_best_param = study.best_params
# %%
# XGBoost with optuna
def objective(trial):

    param_grid_xgb = {
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
        "gamma": trial.suggest_discrete_uniform("gamma", 0.1, 1.0, 0.1),
        "subsample": trial.suggest_discrete_uniform("subsample", 0.5, 1.0, 0.1),
        "colsample_bytree": trial.suggest_discrete_uniform(
            "colsample_bytree", 0.5, 1.0, 0.1
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "random_state": 0,
    }

    model = XGBClassifier(**param_grid_xgb)

    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
xgb_best_param = study.best_params
# %%
# LightGBM with optuna
def objective(trial):

    param_grid_lgb = {
        "num_leaves": trial.suggest_int("num_leaves", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "random_state": 0,
    }

    model = LGBMClassifier(**param_grid_lgb)

    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
lgb_best_param = study.best_params
# %%
# Logistic regression with optuna


def objective(trial):

    param_grid_lr = {"C": trial.suggest_int("C", 1, 100), "random_state": 0}

    model = LogisticRegression(**param_grid_lr)

    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
lr_best_param = study.best_params
# %%
# SVC with optuna


def objective(trial):

    param_grid_svc = {
        "C": trial.suggest_int("C", 50, 200),
        "gamma": trial.suggest_loguniform("gamma", 1e-4, 1.0),
        "random_state": 0,
        "kernel": "rbf",
    }

    model = SVC(**param_grid_svc)

    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
svc_best_param = study.best_params
# %%
# predict with best params
# Evaluate the model with 5-Fold CV / Accuracy
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

rfc_best = RandomForestClassifier(**rfc_best_param)
print("RandomForestClassifier")
print("=" * 20)
scores = cross_validate(rfc_best, X=train_feature, y=train_tagert, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print("=" * 20)

xgb_best = XGBClassifier(**xgb_best_param)
print("XGBClassifier")
print("=" * 20)
scores = cross_validate(xgb_best, X=train_feature, y=train_tagert, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print("=" * 20)

lgb_best = LGBMClassifier(**lgb_best_param)
print("LGBMClassifier")
print("=" * 20)
scores = cross_validate(lgb_best, X=train_feature, y=train_tagert, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print("=" * 20)

lr_best = LogisticRegression(**lr_best_param)
print("LogisticRegression")
print("=" * 20)
scores = cross_validate(lr_best, X=train_feature, y=train_tagert, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print("=" * 20)

svc_best = SVC(**svc_best_param)
print("SVC")
print("=" * 20)
scores = cross_validate(svc_best, X=train_feature, y=train_tagert, cv=kf)
print(f'mean:{scores["test_score"].mean()}, std:{scores["test_score"].std()}')
print("=" * 20)
# %%
from sklearn.ensemble import VotingClassifier

# Prepare classifiers for voting
estimators = [
    ("rfc", RandomForestClassifier(**rfc_best_param)),
    ("xgb", XGBClassifier(**xgb_best_param)),
    ("lgb", LGBMClassifier(**lgb_best_param)),
    ("lr", LogisticRegression(**lr_best_param)),
    ("svc", SVC(**lr_best_param)),
]
voting = VotingClassifier(estimators)

print("VotingClassifier")
print("=" * 20)
scores = cross_validate(voting, X=train_feature, y=train_tagert, cv=kf)
# %%
# RandomForest
rfc_best = RandomForestClassifier(**rfc_best_param)
rfc_best.fit(train_feature, train_tagert)
# XGBoost
xgb_best = XGBClassifier(**xgb_best_param)
xgb_best.fit(train_feature, train_tagert)
# LightGBM
lgb_best = LGBMClassifier(**lgb_best_param)
lgb_best.fit(train_feature, train_tagert)
# LogisticRegression
lr_best = LogisticRegression(**lr_best_param)
lr_best.fit(train_feature, train_tagert)
# SVC
svc_best = SVC(**svc_best_param)
svc_best.fit(train_feature, train_tagert)
# prediction
pred = {
    "rfc": rfc_best.predict(test_feature),
    "xgb": xgb_best.predict(test_feature),
    "lgb": lgb_best.predict(test_feature),
    "lr": lr_best.predict(test_feature),
    "svc": svc_best.predict(test_feature),
}


# %%
# submission
sample = pd.read_csv("../data/sample_submission.csv")
for key, value in pred.items():
    pd.concat(
        [
            pd.DataFrame(sample.PassengerId, columns=["PassengerId"]).reset_index(
                drop=True
            ),
            pd.DataFrame(value, columns=["Transported"]),
        ],
        axis=1,
    ).to_csv(f"../submissions/output_{key}.csv", index=False)
# %%
