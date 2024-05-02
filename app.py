import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import matplotlib.pyplot as plt
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model1 = pipeline("sentiment-analysis")

sentiment_list = []

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
    st.sidebar.write('Chat assistant is a text generation chatbot based on the GPT-2 model.')
    
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
    
    st.write("Sentiment Analysis Result:", sentiment_result[0]['label'])
    
    # Store the sentiment result
    state.sentiment_list.append(sentiment_result[0]['label'])

    result = get_output(user_input)
    if st.button('Run'): 
        st.write(result)
    

    if st.button("Display Sentiments"):
        sentiment_df = pd.DataFrame(state.sentiment_list, columns=["Sentiment"])
        st.write("Sentiment Analysis Results:")
        st.subheader("Bar Graph:")
        st.bar_chart(sentiment_df["Sentiment"].value_counts())
        
        # Plot pie chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_df["Sentiment"].value_counts(), labels=sentiment_df["Sentiment"].unique(), autopct='%1.1f%%')
        ax.axis('equal') 
        st.subheader("Pie Chart:")
        st.pyplot(fig)
