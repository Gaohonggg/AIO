import levenshtein_distance as lvd
import streamlit as st

def read_vocab(file_path):
    with open(file_path, 'r') as f:
        words = set([line.strip().lower() for line in f])
    return words

def main():
    words = read_vocab("C:/Users/Admin/Desktop/PYTHON/AIO/Module 1/exercise4/vocab.txt")
    st.title("Word Correction using Levenshtein Distance Basic")
    word_input = st.text_input("Word: ")
    if st.button("Check"):
        dis = dict()
        for word in words:
            dis[word] = lvd.buildmatrix(word_input,word)
        dis = dict(sorted(dis.items(),key= lambda item : item[1]))
        result = list(dis.keys())[0]
        st.write("Correct word is: ",result)

        col1,col2 = st.columns(2)
        col1.write("Vocabulary: ")
        words = { i : num for i,num in enumerate(words,start=1)}
        col1.write(words)
        col2.write("Distance: ")
        col2.write(dis)

main()