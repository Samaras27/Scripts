import pandas as pd
from pycaret import classification as clf 
import sys
df = pd.read_csv("./data/SUMMARY.csv")
sys.stdout = open('./figures/model_output.txt', 'w')
sys.stderr = sys.stdout
s = clf.setup(df, target = 'tag', session_id = 123)
best = clf.compare_models()
clf.plot_model(best, plot = 'confusion_matrix', save=True)
clf.plot_model(best, plot = 'auc', save=True)
clf.plot_model(best, plot = 'error', save=True)
clf.plot_model(best, plot = 'class_report', save=True)
clf.plot_model(best, plot = 'pr', save=True)
holdout_pred = clf.predict_model(best)