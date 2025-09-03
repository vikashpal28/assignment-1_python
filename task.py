import pandas as pd
import numpy as np
import seaborn as sbs
import sklearn as sklearn
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataSet/student_scores.csv")

print(df)
print("\n")

print("Student Information")
print(df.head(10))

print("\nNo of row and col in the dataSet: ",df.shape)

print("\n")
print(f"the maximum hours is: {df['Hours'].max()} and maximum score is: {df['Scores'].max()}")

print(f"\nthe minimum hours is: {df['Hours'].min()}  and minimum score is: {df['Scores'].min()}")

print(f"\nthe mean hours is: {df['Hours'].mean()}  and mean score is: {df['Scores'].mean()}")

print(f"\nthe mediam hours is: {df['Hours'].median()}  and median score is: {df['Scores'].median()}")

#Train  test Split
x=df[['Hours','Scores']]
y=df['Scores']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
print("Training Shape",x_train.shape)
print("Testing Shape",x_test.shape)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 12))
ax.scatter(x_train["Hours"], x_train["Scores"], color="red", label="Training Data")
ax.scatter(x_test["Hours"], x_test["Scores"], color="blue", label="Testing Data")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks Obtained")
ax.set_title("Study Hours vs Scores")
ax.legend()
plt.show()

