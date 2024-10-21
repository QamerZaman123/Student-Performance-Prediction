from  Project import *
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(max_depth=6)

# Train the model

score = model_train(model)
print(score)