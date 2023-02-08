# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path('./lln/')

df = pd.read_csv('company_income.csv')
# df = df.head(6)
N = df.shape[0]
df = df.sort_values('营业收入（百万美元）')
fig, ax = plt.subplots(1,1, figsize=(8,8), gridspec_kw={'left':0.25, 'right':0.95, 'top':0.98, 'bottom':0.08})
ax.barh(df['公司名称'], df['营业收入（百万美元）'])
ax.set_xlabel('利润（百万美元）', fontsize=18)
ax.set_yticklabels(df['公司名称'].values, fontsize=15)
ax.set_ylim([-0.5,N-0.5])
fig.savefig(path/'company_income.png', dpi=300)