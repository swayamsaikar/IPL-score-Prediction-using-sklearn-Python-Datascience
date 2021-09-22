import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# reading the csv
df = pd.read_csv("ipl.csv")

# removing the unneccessary teams from the dataset and keeping the current consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

# keeps the team names which are present in the abovbe list of bowl and bat  in the dataset
df = df[(df["bat_team"].isin(consistent_teams)
         & df["bowl_team"].isin(consistent_teams))]

# dropping useless columns
columns_drop = ['mid', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_drop, axis=1, inplace=True)

# encoding the bat_team and bowl_team to convert Convert categorical variable into dummy/indicator variables
# it has converted every single values of bat_team and bowl_team into separate columns with different numerical values
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# spliting the data into input factors and the output variables
X = encoded_df.drop(labels=["date", "total", "venue"], axis=1)
Y = encoded_df["total"]

# splitting the whole data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, Y_train)

prediction = classifier.predict(X_test)
Accuracy = r2_score(Y_test, prediction)

print(f"Filtered Data :- {encoded_df.head()}")
# This will give us a list of predictions
print(f"Prediction :- {prediction[0]} score")
# This will give us the accuracy of our model
print(f"Prediction :- {Accuracy}")
