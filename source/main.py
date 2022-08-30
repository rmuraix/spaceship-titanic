# %%
# inport packages
import numpy as np
import pandas as pd
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