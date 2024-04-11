from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import umap

tag_vals = {
    0: "CINS",
    1: "CS",
}
df = pd.read_csv("./data/SUMMARY.csv")
tag = pd.DataFrame(df["tag"], columns=["tag"])
df = df.drop(['tag'], axis=1)
tag = tag.replace({"tag": tag_vals})

############################################ PCA #############################################################
# Calculate PCA with 2 components. PCA is a method that reduces the dimensionality of the data
# by projecting it onto two principal components that capture the most variance in the data.
pca_2d = PCA(n_components=2)
principalComponents_2d = pca_2d.fit_transform(df)
# Convert the principal components into a pandas dataframe for ease of use
principalDf_2d = pd.DataFrame(data=principalComponents_2d, columns=['PC 1', 'PC 2'])
# Concatenate the labels to the dataframe for color-coding the plot
principalDf_2d = pd.concat([principalDf_2d, tag[['tag']]], axis=1)

plt.figure(figsize=(8, 6))
for label, color in zip(tag['tag'].unique(), ["lightskyblue", "lightcoral"]):
    indicesToKeep = principalDf_2d['tag'] == label
    plt.scatter(
        principalDf_2d.loc[indicesToKeep, 'PC 1'], 
        principalDf_2d.loc[indicesToKeep, 'PC 2'], 
        c=color, 
        s=50, 
        edgecolors='gray',  # This adds the gray border around each point
        alpha=0.77  # Adjust the transparency of the points
    )
plt.title('PCA 2D')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(tag['tag'].unique())
plt.savefig("./figures/PCA_2D.pdf", bbox_inches="tight")
plt.show()

######################################## UMAP #########################################
# UMAP is a manifold learning technique for dimension reduction.
# Here, we specify 2 components to project the data into a 2D space.
umap_2d = umap.UMAP(n_components=2)
umap_data_2d = umap_2d.fit_transform(df)

plt.figure(figsize=(8, 6))
for label, color in zip(tag['tag'].unique(), ["lightskyblue", "lightcoral"]):
    indicesToKeep = tag['tag'] == label
    plt.scatter(
        umap_data_2d[indicesToKeep, 0],
        umap_data_2d[indicesToKeep, 1],
        c=color,
        s=50, 
        edgecolors='gray',  # Adding gray edge colors here as well
        alpha=0.77
    )
plt.title('UMAP 2D')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend(tag['tag'].unique())
plt.savefig("./figures/UMAP_2D.pdf", bbox_inches="tight")
plt.show()

######################################## t-SNE #########################################################
# t-SNE is a technique for dimensionality reduction that is particularly well suited for
# the visualization of high-dimensional datasets.
tsne_2d = TSNE(n_components=2, random_state=0)
tsne_data_2d = tsne_2d.fit_transform(df.copy())
plt.figure(figsize=(8, 6))
for label, color in zip(tag['tag'].unique(), ["lightskyblue", "lightcoral"]):
    indicesToKeep = tag['tag'] == label
    plt.scatter(
        tsne_data_2d[indicesToKeep, 0],
        tsne_data_2d[indicesToKeep, 1],
        c=color,
        s=50, 
        edgecolors='gray',  # And here
        alpha=0.77
    )
plt.title('T-SNE 2D')
plt.xlabel('T-SNE 1')
plt.ylabel('T-SNE 2')
plt.legend(tag['tag'].unique())
plt.savefig("./figures/t_SNE_2D.pdf", bbox_inches="tight")
plt.show()

#################################### PCA 3D ##########################################3
# Assuming 'df' is your DataFrame with the data and 'labels' is the Series or list with the labels
pca_3d = PCA(n_components=3)
principalComponents_3d = pca_3d.fit_transform(df)
principalDf_3d = pd.DataFrame(data=principalComponents_3d, columns=['PC 1', 'PC 2', 'PC 3'])

# Assuming 'labels' is your labels list or Series that is aligned with 'df'
principalDf_3d['tag'] = tag
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
unique_labels = np.unique(tag)
colors = ["lightcoral", "lightskyblue"]

for label, color in zip(unique_labels, colors):
    indicesToKeep = principalDf_3d['tag'] == label
    ax.scatter(
        principalDf_3d.loc[indicesToKeep, 'PC 1'],
        principalDf_3d.loc[indicesToKeep, 'PC 2'],
        principalDf_3d.loc[indicesToKeep, 'PC 3'],
        c=[color],
        s=50,
        edgecolors='gray',  # Gray border around each point
        alpha=0.7,
        label=label
    )

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3', labelpad=-1.5)
ax.set_title('PCA 3D')
ax.legend()
plt.savefig("./figures/PCA_3D.pdf", bbox_inches="tight")
plt.show()

################################# 3D UMAP##############################
# Initialize UMAP. The random_state parameter ensures reproducibility
umap_3d = umap.UMAP(n_components=3, random_state=42)

# Compute UMAP representation
umap_result_3d = umap_3d.fit_transform(df)

# Create a DataFrame for the UMAP result
umap_df_3d = pd.DataFrame(umap_result_3d, columns=['UMAP 1', 'UMAP 2', 'UMAP 3'])

# Add the labels as a column to this DataFrame
umap_df_3d['tag'] = tag["tag"].values
# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Obtain the unique labels for plotting
unique_labels = tag['tag'].unique()

# Create a color map
colors = ["lightskyblue", "lightcoral"]

# Plot each cluster
for label, color in zip(unique_labels, colors):
    indices = umap_df_3d['tag'] == label
    ax.scatter(
        umap_df_3d.loc[indices, 'UMAP 1'],
        umap_df_3d.loc[indices, 'UMAP 2'],
        umap_df_3d.loc[indices, 'UMAP 3'],
        c=[color],
        edgecolors='gray',  # This adds the gray border around the points
        s=50,
        alpha=0.7,
        label=label
    )

ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3', labelpad=-1.3)
ax.legend()
ax.set_title('3D UMAP Representation')
plt.savefig("./figures/UMAP_3D.pdf", bbox_inches="tight")
plt.show()

################################## 3D t-SNE ########################################
# Assuming 'labels' is a Pandas series or a list that corresponds to the labels of the rows in 'df'.
tsne_3d = TSNE(n_components=3, random_state=42)
tsne_data_3d = tsne_3d.fit_transform(df)

# Convert the t-SNE output to a DataFrame for easier manipulation
tsne_df_3d = pd.DataFrame(tsne_data_3d, columns=['TSNE-1', 'TSNE-2', 'TSNE-3'])

# Add the labels to this DataFrame
tsne_df_3d['tag'] = tag
color_map = {
    'CS': 'lightcoral',
    'CINS': 'lightskyblue'
    # Adjust the keys and colors to your specific labels
}

# Create a new figure for plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Group the data by labels and plot each group with its color
for label, group in tsne_df_3d.groupby('tag'):
    ax.scatter(
        group['TSNE-1'], group['TSNE-2'], group['TSNE-3'],
        color=color_map[label],  # Use the color defined in color_map
        label=label,
        edgecolor='gray',  # Black edges around the dots for better visibility
        s=50,           # Size of the dots
        alpha=0.7       # Transparency of the dots
    )

# Set labels for axes
ax.set_xlabel('T-SNE 1')
ax.set_ylabel('T-SNE 2')
ax.set_zlabel('T-SNE 3', labelpad=-1.5)

# Title for the plot
ax.set_title('3D T-SNE Visualization')

# Legend to show labels
ax.legend()
plt.savefig("./figures/t_SNE_3D.pdf", bbox_inches="tight")
# Show plot
plt.show()