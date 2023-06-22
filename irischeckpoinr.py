import streamlit as st
import sklearn
import sklearn.datasets
import sklearn.ensemble
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
iris_X,iris_y = sklearn.datasets.load_iris(return_X_y = True)
iris= sklearn.datasets.load_iris()
clf=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 10 decision trees
clf.fit(iris_X,iris_y)

st.title("welcome to iris dataset")
st.header("let's train and test the model")


sepal_length = st.slider("Sepal Length", float(iris_X[:, 0].min()), float(iris_X[:, 0].max()), float(iris_X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(iris_X[:, 1].min()), float(iris_X[:, 1].max()), float(iris_X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(iris_X[:, 2].min()), float(iris_X[:, 2].max()), float(iris_X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(iris_X[:, 3].min()), float(iris_X[:, 3].max()), float(iris_X[:, 3].mean()))


if(st.button("predict")):
    inputs= [[sepal_length, sepal_width, petal_length, petal_width]]
    pred = clf.predict(inputs)
    st.write("the type of the iris flower is ", pred)

    target_names = iris.target_names
    st.write(f"Predicted Iris Flower Type: {target_names[pred[0]]}")