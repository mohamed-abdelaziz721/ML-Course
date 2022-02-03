# # Visualizing the tree.
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier
# from IPython.display import Image
# from IPython.core.display import HTML 

# df = pd.read_csv("cardio_train.csv", delimiter= ";")
# df.drop('id', inplace=True, axis=1)
# df['age'] = np.ceil(df['age']/365)
# # select the wanted column
# df_text_genre = df[list(df)]
# train, test = train_test_split(df_text_genre, test_size = 0.1, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(df_text_genre, df.cardio , test_size = 0.1, random_state=42)

# tree_model = DecisionTreeClassifier(max_depth=2)
# tree_model.fit(X_train.age.to_frame(), X_train.cardio)
# X_train['age_tree']=tree_model.predict_proba(X_train.age.to_frame())[:,1] 


# with open("tree_model2.gv", "w") as f:
#     f = tree.export_graphviz(tree_model, out_file=f)
    

# PATH = "tree_visualisation.png"
# Image(filename = PATH , width=1000, height=1000)


# # dot_path= "tree_model2.gv"
# # s = Source.from_file(dot_path)
# # s.view()

# # dot_data = tree.export_graphviz(tree_model, filled=True, rounded=True,  
# #                                   special_characters=True,out_file=None,)
# # graph = graphviz.Source(dot_data)
# # graph

# # graph.format = "png"
# # graph.render("age_tree_discretized")



import os
os.sys.path.append(r'C:\Users\Mohamed Abdelaziz\AppData\Local\Programs\Python\Python39\Lib\site-packages\graphviz')

# python -c "import sys; print(sys.path)"



tree_model = DecisionTreeClassifier(max_depth=2)
tree_model.fit(X_train.age.to_frame(), X_train.cardio)
X_train['age_tree']=tree_model.predict_proba(X_train.age.to_frame())[:,1] 
X_train.head(10)

X_train.groupby('age_tree').age_tree.value_counts().unstack(0).plot.barh()

age_tree_groups = X_train.groupby('age_tree')
age_tree_groups.first()

height_tree_groups.size()

pd.concat( [X_train.groupby(['age_tree'])['age'].min(),
            X_train.groupby(['age_tree'])['age'].max()], axis=1)

X_train.age_tree.unique()


# Check the relationship between the discretized variable age_tree and the target cardio.
fig = plt.figure()
fig = X_train.groupby(['age_tree'])['cardio'].mean().plot()
fig.set_title('Monotonic relationship between discretised age and cardio')
fig.set_ylabel('cardio')

#  That plot suggests that age_tree seems like a good predictor of the target variable cardio .

# Visualizing the tree.
from sklearn import tree
from graphviz import Source

dot_file = "age_tree_model.gv"
with open(dot_file, "w") as f:
    f = tree.export_graphviz(tree_model, out_file=f)
    
s = Source.from_file(dot_file, format='png')
s.view()    
    
from IPython.display import Image
Image(filename=f'{dot_file}.png')     

['age','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']




X_train.gender.replace({1: "female", 2: "male"}, inplace=True)
X_train.cholesterol.replace({1: "normal", 2: "above normal", 3: "well above normal"}, inplace=True)
X_train.gluc.replace({1: "normal", 2: "above normal", 3: "well above normal"}, inplace=True)
X_train.smoke.replace({0: "doesn't smoke", 1: "smokes"}, inplace=True)
X_train.alco.replace({0: "doesn't drink", 1: "drinks"}, inplace=True)
X_train.active.replace({0: "not active", 1: "active"}, inplace=True)
X_train.cardio.replace({0: "absence", 1: "presence"}, inplace=True)
X_train.head(5)