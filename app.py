import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import matplotlib.pyplot as plt
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model1 = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_state():
    return st.session_state

def get_output(line):
    output = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    return output(line, max_length=100, num_return_sequences=1)

# Get the session state
state = get_state()

if 'sentiment_list' not in state:
    state.sentiment_list = []

if __name__ == '__main__':
    st.sidebar.title('Chat assistant')
    st.sidebar.subheader('About')
    st.sidebar.write('Chat assistant is a text generation chatbot based on the GPT-2 model. It also do sentiment analysis on user inputs.')
    st.sidebar.write('5 stars = Highly Positive')
    st.sidebar.write('4 stars = Positive')
    st.sidebar.write('3 stars = Neutral')
    st.sidebar.write('2 stars = Negative')
    st.sidebar.write('1 star = Highly Negative')
    
    st.sidebar.subheader('How to Use')
    st.sidebar.write('Enter your message as input, then click "Run" to generate output.')
    
    st.sidebar.subheader('Contributors')
    st.sidebar.write('Contributor: Aditya Sharma')
 

    st.title('Chat Assistant')
    st.write('Transformer Architecture: {}'.format('gpt-2'))
    st.subheader("Input")
    user_input = st.text_area('', height=25)
    # Perform sentiment analysis on the user input
    sentiment_result = model1(user_input)
    
    # Extract sentiment label from the result
    sentiment_label = sentiment_result[0]['label']
    
    st.write("Sentiment Analysis Result:", sentiment_label)
    
    # Store the sentiment result
    state.sentiment_list.append(sentiment_label)

    result = get_output(user_input)
    if st.button('Run'): 
        st.write(result)
    

    if st.button("Display Sentiments"):
        sentiment_df = pd.DataFrame(state.sentiment_list, columns=["Sentiment"])
        st.write("Sentiment Analysis Results:")
        st.subheader("Bar Graph:")
        st.bar_chart(sentiment_df["Sentiment"].value_counts())
        
        # Count occurrences of each sentiment label
        sentiment_counts = sentiment_df["Sentiment"].value_counts()
    
        
        st.write("Number of Highly Positive Sentiments (5 stars):", sentiment_df["Sentiment"].str.count("5 stars").sum())
        st.write("Number of Positive Sentiments (4 stars):", sentiment_df["Sentiment"].str.count("4 stars").sum())
        st.write("Number of Neutral Sentiments (3 stars):", sentiment_df["Sentiment"].str.count("3 stars").sum())
        st.write("Number of Negative Sentiments (2 stars):", sentiment_df["Sentiment"].str.count("2 stars").sum())
        st.write("Number of Highly Negative Sentiments (1 star):", sentiment_df["Sentiment"].str.count("1 star").sum())
        
        # Plot pie chart
        fig, ax = plt.subplots()
        ax.pie([sentiment_df["Sentiment"].str.count("5 stars").sum(), sentiment_df["Sentiment"].str.count("4 stars").sum(),
                sentiment_df["Sentiment"].str.count("3 stars").sum(), sentiment_df["Sentiment"].str.count("2 stars").sum(),
                sentiment_df["Sentiment"].str.count("1 star").sum()], 
               labels=["Highly Positive", "Positive", "Neutral", "Negative", "Highly Negative"], autopct='%1.1f%%')
        ax.axis('equal') 
        st.subheader("Pie Chart:")
        st.pyplot(fig)
