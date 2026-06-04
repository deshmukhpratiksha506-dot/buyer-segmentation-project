import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Title
st.title("Machine Learning Based Buyer Segmentation")

# Load Dataset
df = pd.read_csv("final_clustered_data.csv")

# Sidebar Filters
st.sidebar.header("Filters")

country = st.sidebar.multiselect(
    "Select Country",
    options=df['country'].unique(),
    default=df['country'].unique()
)

region = st.sidebar.multiselect(
    "Select Region",
    options=df['region'].unique(),
    default=df['region'].unique()
)

client_type = st.sidebar.multiselect(
    "Select Client Type",
    options=df['client_type'].unique(),
    default=df['client_type'].unique()
)

# Filter Data
filtered_df = df[
    (df['country'].isin(country)) &
    (df['region'].isin(region)) &
    (df['client_type'].isin(client_type))
]

# Show Dataset
st.subheader("Filtered Dataset")
st.write(filtered_df)

# Cluster Distribution
st.subheader("Buyer Segment Distribution")

fig, ax = plt.subplots(figsize=(8,5))

sns.countplot(x='Cluster', data=filtered_df, ax=ax)

st.pyplot(fig)

# Satisfaction Score by Cluster
st.subheader("Satisfaction Score by Cluster")

fig2, ax2 = plt.subplots(figsize=(8,5))

sns.boxplot(
    x='Cluster',
    y='satisfaction_score',
    data=filtered_df,
    ax=ax2
)

st.pyplot(fig2)

# Age Distribution
st.subheader("Age Distribution")

fig3, ax3 = plt.subplots(figsize=(8,5))

sns.histplot(filtered_df['Age'], bins=20, ax=ax3)

st.pyplot(fig3)

# Cluster Insights
st.subheader("Cluster Insights")

cluster_insights = filtered_df.groupby('Cluster').mean(numeric_only=True)

st.write(cluster_insights)