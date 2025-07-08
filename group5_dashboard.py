import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import statsmodels.api as sm
# Load the dataset
data = pd.read_csv('CLV_revised.csv')

# App Title and Description

'''
# ADTA 5410 - Group 5

#### This dashboard aims to gain a deeper understanding of what drives long-term customer value, beyond the amount spent.

### Key Observations:
- `total_spend_12m` contains a correlation coefficient of 1 with `clv_3yr_usd`, indicating a powerful baseline predictor.
- Although `total_spend_12m` contains a high correlation coefficient, when outliers are removed from `clv_3yr_usd` spending tiers show diminishing returns in prediction power.

### *Problem Statement:* Among high-spending customers, do certain behaviors like website engagement and return history help explain why some customers are more valuable than others?

### Behavioral Variables to Explore:
- `sessions_90d` - how often a customer is returning to the website
- `returns_12m` - how many returns ocurred in the last 12 months
- `email_open_rate` - Frequency of opened emails (%)

#### *Objective:* We will compare high spenders based on these behavioral traits to observe for meaningful differences in their 3-year Customer Lifetime Value to determine which variables provide stronger predictive power to help build smarter, more personalized strategies.
'''
    


'''
# Exploratory Data Analysis
'''


# Sidebar Filters
st.sidebar.header("Filter Options")   # give a title for the sidebar menu


# Sidebar Filters for Numerical Variables
sessions_range = st.sidebar.slider("Website Sessions", int(data['sessions_90d'].min()), int(data['sessions_90d'].max()), (1, 41))
return_range = st.sidebar.slider("Number of Returns", int(data['returns_12m'].min()), int(data['returns_12m'].max()), (1, 12))
# create a slider, and tentatively show a range from 5 to 30



# Sidebar Filters for Categorical Variables
region = st.sidebar.multiselect("Region", options=data['region_name'].unique(), default=data['region_name'].unique())
pref_device = st.sidebar.multiselect("Preferred Device", options=data['preferred_device'].unique(), default=data['preferred_device'].unique())
prem_member = st.sidebar.multiselect("Premium Member", options=data['is_premium_member'].unique(), default=data['is_premium_member'].unique())


# Filter data based on selections
filtered_data = data[
    (data['sessions_90d'].between(*sessions_range)) &
    (data['returns_12m'].between(*return_range)) &
    (data['region_name'].isin(region)) &
    (data['preferred_device'].isin(pref_device)) &
    (data['is_premium_member'].isin(prem_member))
]


# Show filtered data if user selects the option
if st.sidebar.checkbox("Show Filtered Data"):
    st.write(filtered_data)

'''
_Data relative to 06-13-2025_
'''
st.write(f"Total dataset: {len(data):,}")
st.write(f"Current filtered dataset: {len(filtered_data):,}")


#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~ Baseline Data ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
'''
## Pair Plot of Customer Lifetime Value vs Behavioral Traits.
The pairplot shows non-linear relationships between spend and behavioral traits.
'''

# Set columns for pair plot
cols = ['clv_3yr_usd', 'total_spend_12m', 'sessions_90d', 'returns_12m', 'email_open_rate']
pplot_data = filtered_data[cols]
# Plot Pair Plot
pplot = sns.pairplot(pplot_data, diag_kind='kde', corner=True)
pplot.fig.suptitle("Pair Plot: CLV vs Spend and Behavioral Traits", y=1.02)
st.pyplot(pplot.fig)




#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~ INSIGHT 1 ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
'''
## Heatmap Plot of Correlation Coefficients
The heatmap further shows that most variables do not have a strong linear relationship with `clv_3yr_usd`, although `total_spend_12m` has a correlation coefficient of 1. May features contain coefficients close to zero.
'''

corr = filtered_data[cols].corr()
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
ax.set_title("Heatmap of Correlation Coefficients")
st.pyplot(fig)




#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~ INSIGHT 2 ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
'''
## Box Plot of customers grouped by spending tiers.
The box plot displays how customers in the high tier of spending contains a large variability in a 3 year projected spend. Graph A removed outliers from the target variable `clv_3yr_usd` and shows how spend tiers reflect real spend behavior. This shows that although high spenders exist, their projected CLV is volatile. Graph B shows the data with outliers to display full value distribution. 
'''
# Remove outliers for Graph A
df_MAD = filtered_data.loc[filtered_data['MAD_clv_3yr_usd'] == False].copy()

category_order = ['Low', 'Medium', 'High']

df_MAD['spend_tier'] = pd.Categorical(df_MAD['spend_tier'], categories=category_order, ordered=True)

fig, ax = plt.subplots(figsize=(10,6))
sns.boxplot(data=df_MAD, x='spend_tier', y='clv_3yr_usd', ax=ax)
ax.set_title("Graph A: CLV by Spend Tier (Outliers Removed with MAD)")
ax.set_ylabel("3-Year Customer Lifetime Value")
ax.set_xlabel("Spend Tier")
st.pyplot(fig)

filtered_data['spend_tier'] = pd.Categorical(filtered_data['spend_tier'], categories=category_order, ordered=True)

if st.checkbox("Show CLV with Outliers"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=filtered_data, x='spend_tier', y='clv_3yr_usd', ax=ax)
    ax.set_title("Graph B: CLV by Spend Tier")
    ax.set_ylabel("3-Year Customer Lifetime Value")
    ax.set_xlabel("Spend Tier")
    st.pyplot(fig)



#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~ INSIGHT 3 ~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#
'''
## Violin plot of website sessions in the last 90 days
The violin plot shows the density shapes between spending tiers compared to recent 90-day sessions. Although the shapes of each tier are similar, it is seen that the low spending tier has a higher session count than the high spending tier. This reveals that more sessions does not equate to more spend, and can develop behavioral shopping characteristics of deal hunters or price conscious consumers.
'''
fig, ax = plt.subplots(figsize=(10,6))
sns.violinplot(data=filtered_data, x='spend_tier', y='sessions_90d', ax=ax)
ax.set_title("Spending Tier vs Recent Website Visits (90d)")
ax.set_ylabel("Recent Website Visits")
ax.set_xlabel("Spend Tier")
st.pyplot(fig)

