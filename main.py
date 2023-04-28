from naive_bayes import naive_bayes as o
import streamlit as st
st.header('Prediction using Naive Bayes Classifier')
input = st.text_area("Please enter the text", value="")
l=[]
l.append(input)
if st.button("Predict"):
    
    
    vec = o.vector.transform(l).toarray()
    k=str(list(o.naivebayes.predict(vec))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2', 'SPORTS').replace('3','ENTERTAINMENT').replace('4','POLITICS')
    st.write(k)