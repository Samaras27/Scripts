import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

TAG_VALS_CAT = {
    1: "CS",
    0: "CINS",
}

TAG_VALS_NUM = {
    "CS": 1,
    "CINS": 0,
}
df = pd.read_csv("./data/SUMMARY.csv")
df["tag"] = df["tag"].replace(TAG_VALS_CAT)


################################### BOX PLOTS #################################################

# Assuming 'df' is your DataFrame and the last column is 'category'
# You previously counted the occurrences of each category and reordered them
category_counts = df['tag'].value_counts().reindex(['CINS', 'CS'])

# Define the colors for each category
colors = ['#fbb4b9', '#c51b8a']  # Colors for 'control' and 'case', respectively

# Creating the bar plot with specified bar width and colors
bar_width = 0.4  # Make the bars thinner by setting width < 0.8 (default is 0.8)
plt.bar(category_counts.index, category_counts.values, width=bar_width, color=colors)

# Adding the title
plt.title('Number of Samples by Category')

# Labeling the axes
plt.xlabel('Category')
plt.ylabel('Number of Samples')

# Optional: Add text labels on top of the bars
for i, count in enumerate(category_counts.values):
    plt.text(i, count, str(count), ha='center', va='bottom')

plt.savefig("./figures/box_plot.pdf")
# Show the plot
plt.show()

################################### VIOLIN PLOTS #################################################

# Assuming 'df' is your DataFrame and 'tag' is the name of the column with categorical data
# Store the 'tag' column in its own DataFrame
tag_df = df[['tag']].copy()

# Drop the 'tag' column from the main DataFrame
df_numerical = df.drop('tag', axis=1)

# Initialize the StandardScaler
scaler = StandardScaler()

# Perform scaling on the numerical data
df_scaled_numerical = pd.DataFrame(
    scaler.fit_transform(df_numerical),
    columns=df_numerical.columns,
    index=df_numerical.index  # This preserves the original indexing
)

# Concatenate the scaled numerical data with the 'tag' column
df_scaled = pd.concat([df_scaled_numerical, tag_df], axis=1)


# Assuming df is your pre-existing DataFrame and melted_df is created as shown
melted_df = pd.melt(df_scaled, id_vars="tag", value_vars=[col for col in df_scaled.columns if col != "condition"],
                    var_name="feature", value_name="value")

# Assuming df is your pre-existing DataFrame and melted_df is created as shown
melted_df = pd.melt(df, id_vars="tag", value_vars=[col for col in df.columns if col != "condition"],
                    var_name="feature", value_name="value")
fig, ax = plt.subplots(figsize =(12, 8))  
plot_palette = {'CINS':'#fbb4b9','CS':'#c51b8a'}
sns.violinplot(data=melted_df, 
                  x='value',
                  y='feature',
                  hue='tag', 
                  split=True,
                  inner='quartile', 
                  palette=plot_palette,
                  density_norm="count",
                  )
plt.xticks(rotation=0)
plt.title("Violin Plot CINS vs. CS features")
plt.grid(axis='x')

plt.savefig("./figures/violin_plot.pdf", bbox_inches="tight")
# Show the plot
plt.show()

################################### CORRELATION HEATMAP #################################################
df["tag"] = df["tag"].replace(TAG_VALS_NUM)
# Compute pairwise correlation of columns, 
# excluding NA/null values.
corr_matrix = df.corr(method='spearman')

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20,10))

# Draw the heatmap with the mask 
# and correct aspect ratio
sns.heatmap(data=corr_matrix,
            cmap='RdPu', annot=False, 
            square=True, 
            linewidths=.5, fmt=".1f")
plt.savefig("./figures/corr_heatmap.pdf", bbox_inches="tight")
# Show the plot
plt.show()
