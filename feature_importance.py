import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Function to assign a rank to each element in a list with the first element being the most important.
def rank_list(lst:list)->dict:
    # Returns a dictionary comprehension where each list element is a key and its rank is a value.
    # The rank is calculated as the length of the list minus one minus the element's index.
    return {elm:len(lst) - 1 - i for i, elm in enumerate(lst)}

# Function to perform Borda count aggregation for a list of ranked lists.
def borda_aggregation(loflists: list[list]) -> dict:
    # Convert each individual list into a dictionary of ranks.
    list_ranks = [rank_list(l) for l in loflists]
    # Create a set of all unique elements across all the lists.
    feature_set = {i for i in [el for nl in loflists for el in nl]}
    # Return a dictionary where each element's score is the sum of its ranks across all the lists.
    return {e:sum([lr.get(e, 0) for lr in list_ranks]) for e in feature_set}

# Function to create a sorted DataFrame from a dictionary of results.
def create_sorted_df(result:dict):
    # Create a DataFrame from the dictionary.
    df = pd.DataFrame(list(result.items()), columns=['Feature', 'Borda Rank'])
    # Sort the DataFrame based on the 'Borda Rank' column in descending order.
    df.sort_values(by='Borda Rank', ascending=False, inplace=True)
    # Reset the DataFrame's index and drop the old index.
    df.reset_index(drop=True, inplace=True)
    # Return the sorted DataFrame.
    return df

# Function to create a DataFrame from a list of lists using Borda count aggregation.
def borda_df(loflists: list[list]):
    # Aggregate the lists into a Borda count dictionary.
    borda_dict = borda_aggregation(loflists)
    # Create and return a sorted DataFrame from the Borda count dictionary.
    return create_sorted_df(borda_dict)

# Function to create a DataFrame showing the importance of features from a model.
def feature_importance_df_creation(model, model_label, feature_names):
    # Create a DataFrame from the model's feature importances.
    importance_df = pd.DataFrame(model.feature_importances_)
    # Rename the column to the provided model label.
    importance_df = importance_df.rename(columns= {0: model_label})
    # Add the feature names as a column in the DataFrame.
    importance_df["feature_name"] = feature_names
    # Sort the DataFrame based on the importance scores in descending order.
    importance_df = importance_df.sort_values(by=model_label, ascending=False)
    # Return the sorted DataFrame.
    return importance_df

df = pd.read_csv("./data/SUMMARY.csv")

tag = pd.DataFrame(df, columns=["tag"])
df = df.drop(columns=['tag'])

xgbclass = XGBClassifier(eval_metric='logloss', verbosity = 0, importance_type='gain', silent=True)
cbc = CatBoostClassifier(iterations=100, verbose=0)
lgbm = LGBMClassifier(importance_type='split')
xgbclass.fit(df, tag["tag"])
cbc.fit(df, tag["tag"])
lgbm.fit(df, tag["tag"])

xgb_importance = feature_importance_df_creation(xgbclass, "XgBoost_importance", list(df.columns))
cbc_importance = feature_importance_df_creation(cbc, "CatBoost_importance", list(df.columns))
lgbm_importance = feature_importance_df_creation(lgbm, "LightGBM_importance", list(df.columns))

xgb_importance

xgboost_imp_lst = list(xgb_importance["feature_name"])
catboost_imp_lst = list(cbc_importance["feature_name"])
lgbm_imp_lst = list(lgbm_importance["feature_name"])

xgboost_imp_lst

borda_results = borda_df([xgboost_imp_lst, catboost_imp_lst, lgbm_imp_lst])
borda_results


# Display the XGBoost feature importance DataFrame
print("XGBoost Feature Importance:")
print(xgb_importance)

# Save XGBoost feature importance DataFrame to PDF
pdf_filename = "./figures/xgb_feature_importance.pdf"
with PdfPages(pdf_filename) as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=xgb_importance, x="XgBoost_importance", y="feature_name", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    pdf.savefig(fig, bbox_inches="tight")

# Display the XGBoost feature importance list
print("XGBoost Feature Importance List:")
print(xgboost_imp_lst)

# Save XGBoost feature importance list to a text file
with open("./figures/xgb_feature_importance_list.txt", "w") as f:
    for item in xgboost_imp_lst:
        f.write("%s\n" % item)


plt.figure(figsize=(8, 6))
ax = sns.scatterplot(data=borda_results, x="Borda Rank", y="Feature", s=100, color="steelblue", label=" ")
x_start = borda_results["Borda Rank"].min()
ax.set_xlim(left=x_start)
ax.set_facecolor("whitesmoke")
for _, row in borda_results.iterrows():
    ax.plot([x_start, row["Borda Rank"]], [row["Feature"], row["Feature"]], color="steelblue", linewidth=0.5)
ax.set_title("Borda Consensus Feature Importance")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
ax.legend_.remove()
plt.tight_layout()
plt.savefig("./figures/borda_feature_impotance.pdf", bbox_inches="tight")

# Save Borda results to a text file
with open("./figures/borda_results.txt", "w") as f:
    f.write("Feature\tBorda Rank\n")
    for _, row in borda_results.iterrows():
        f.write(f"{row['Feature']}\t{row['Borda Rank']}\n")
        
plt.show()