# %%
# inport packages
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
# %%
# import data
df = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
# %%
# Assume names and IDs do not affect the results and remove them
df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
df_test.drop(['Name', 'PassengerId'], axis=1, inplace=True)
# %%
# Handling missing values

# Fill in the missing values in the CryoSleep column with the most common value
df['CryoSleep'].fillna(df['CryoSleep'].mode()[0], inplace=True)
df_test['CryoSleep'].fillna(df_test['CryoSleep'].mode()[0], inplace=True)

# Fill in the missing values in the Cabin column with the most common value
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
df_test['Cabin'].fillna(df_test['Cabin'].mode()[0], inplace=True)

# Fill in the missing values in the Destination column with the median value
df['Destination'].fillna(df['Destination'].mode()[0], inplace=True)
df_test['Destination'].fillna(df_test['Destination'].mode()[0], inplace=True)

# Fill in the missing values in the Age column with the median value
df['Age'].fillna(df['Age'].median(), inplace=True)
df_test['Age'].fillna(df_test['Age'].median(), inplace=True)

# Fill in the missing values in the HomePlanet column with the most common value
df['HomePlanet'].fillna(df['HomePlanet'].mode()[0], inplace=True)
df_test['HomePlanet'].fillna(df_test['HomePlanet'].mode()[0], inplace=True)

# Fill in the missing values in the VIP column with the most common value
df['VIP'].fillna(df['VIP'].mode()[0], inplace=True)
df_test['VIP'].fillna(df_test['VIP'].mode()[0], inplace=True)

# Fill in the missing values in the RoomService column with the most common value
df['RoomService'].fillna(df['RoomService'].mode()[0], inplace=True)
df_test['RoomService'].fillna(df_test['RoomService'].mode()[0], inplace=True)

# Fill in the missing values in the FoodCourt column with the most common value
df['FoodCourt'].fillna(df['FoodCourt'].mode()[0], inplace=True)
df_test['FoodCourt'].fillna(df_test['FoodCourt'].mode()[0], inplace=True)

# Fill in the missing values in the ShoppingMall column with the most common value
df['ShoppingMall'].fillna(df['ShoppingMall'].mode()[0], inplace=True)
df_test['ShoppingMall'].fillna(df_test['ShoppingMall'].mode()[0], inplace=True)

# Fill in the missing values in the Spa column with the most common value
df['Spa'].fillna(df['Spa'].mode()[0], inplace=True)
df_test['Spa'].fillna(df_test['Spa'].mode()[0], inplace=True)

# Fill in the missing values in the VRDeck column with the most common value
df['VRDeck'].fillna(df['VRDeck'].mode()[0], inplace=True)
df_test['VRDeck'].fillna(df_test['VRDeck'].mode()[0], inplace=True)
# %%
# Split cabin by '/'
df["deck"]=df["Cabin"].str.split("/", expand=True)[0]
df["num"]=df["Cabin"].str.split("/",expand=True)[1].astype("float")
df["side"]=df["Cabin"].str.split("/",expand=True)[2]
df_test["deck"]=df_test["Cabin"].str.split("/", expand=True)[0]
df_test["num"]=df_test["Cabin"].str.split("/", expand=True)[1].astype("float")
df_test["side"]=df_test["Cabin"].str.split("/", expand=True)[2]

df.drop(['Cabin'], axis=1, inplace=True)
df_test.drop(['Cabin'], axis=1, inplace=True)
# %%
df.info()
# %%
# Convert categorical data to numerical dataa
# one-hot encoding
df = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'deck', 'side'], sparse=True)
df_test = pd.get_dummies(df_test, columns=['HomePlanet', 'CryoSleep', 'VIP', 'Destination', 'deck', 'side'], sparse=True)
# %%
X_train = df.drop("Transported", axis=1)
y_train = df["Transported"]
X_test  = df_test
X_train.shape, y_train.shape, X_test.shape
# %%
# Rundom forest with optuna
cv = 5

def objective(trial):
    
    param_grid_rfc = {
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        'min_samples_split': trial.suggest_int("min_samples_split", 7, 15),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        'max_features': trial.suggest_int("max_features", 3, 10),
        "random_state": 0
    }

    model = RandomForestClassifier(**param_grid_rfc)
    
    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
rfc_best_param = study.best_params
# %%
# XGBoost with optuna
def objective(trial):
    
    param_grid_xgb = {
        'min_child_weight': trial.suggest_int("min_child_weight", 1, 5),
        'gamma': trial.suggest_discrete_uniform("gamma", 0.1, 1.0, 0.1),
        'subsample': trial.suggest_discrete_uniform("subsample", 0.5, 1.0, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform("colsample_bytree", 0.5, 1.0, 0.1),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        "random_state": 0
    }

    model = XGBClassifier(**param_grid_xgb)
    
    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
xgb_best_param = study.best_params
# %%
# LightGBM with optuna
def objective(trial):
    
    param_grid_lgb = {
        'num_leaves': trial.suggest_int("num_leaves", 3, 10),
        'learning_rate': trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
        'max_depth': trial.suggest_int("max_depth", 3, 10),
        "random_state": 0
    }

    model = LGBMClassifier(**param_grid_lgb)
    
    # Evaluate the model with 5-Fold CV / Accuracy
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_validate(model, X=X_train, y=y_train, cv=kf)
    # Minimize, so subtract score from 1.0
    return scores['test_score'].mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
lgb_best_param = study.best_params