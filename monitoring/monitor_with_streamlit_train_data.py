import streamlit as st
import pandas as pd

st.markdown("# Car Evaluation Dataset Monitor")
st.sidebar.markdown("# Car Evaluation Dataset Monitor")

@st.cache_data
def get_raw_data():
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    return df_train, df_test

header = st.container()
dataset = st.container()
plot_area_code = st.container()

with header:
    st.title('Monitoring some elements')
    st.text("Inspect the processed train/test splits stored in train.csv and test.csv")

with dataset:
    st.header("Dataset snapshot")
    df_train, df_test = get_raw_data()
    st.subheader("Train sample")
    st.write(df_train.head(20))
    st.subheader("Test sample")
    st.write(df_test.head(20))


with plot_area_code:
    st.header("Categorical value counts")
    categorical_columns = df_train.columns.tolist()
    selected_column = st.selectbox(
        "Choose a column to visualize",
        categorical_columns,
        index=categorical_columns.index("buying") if "buying" in categorical_columns else 0,
    )
    st.bar_chart(df_train[selected_column].value_counts())

# streamlit run monitor_with_streamlit_train_data.py