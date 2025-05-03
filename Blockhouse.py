#!/usr/bin/env python
# coding: utf-8

# ## Block House Quantitative Strategist Internship 2025 Assignment

# ### Name: Aditya Rajendra Madkar

# ### ORDER FLOW IMBALANCE

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\Blockhouse\first_25000_rows.csv")
data.head()


# ### Step 1

# In[6]:


def compute_best_level_ofi(df):
    ofi_best = []
    prev_bid = prev_ask = prev_bid_size = prev_ask_size = None

    for _, row in df.iterrows():
        bid = row['bid_px_00']
        ask = row['ask_px_00']
        bid_size = row['bid_sz_00']
        ask_size = row['ask_sz_00']

        # Skip computation if any prior value is None (i.e., first iteration)
        if None in [prev_bid, prev_ask, prev_bid_size, prev_ask_size]:
            ofi_best.append(0)
        else:
            bid_contrib = (
                bid_size if bid > prev_bid else
                (bid_size - prev_bid_size if bid == prev_bid else -prev_bid_size)
            )
            ask_contrib = (
                -ask_size if ask < prev_ask else
                (prev_ask_size - ask_size if ask == prev_ask else prev_ask_size)
            )
            ofi_best.append(bid_contrib + ask_contrib)

        # Update previous values
        prev_bid, prev_ask = bid, ask
        prev_bid_size, prev_ask_size = bid_size, ask_size

    df['OFI_best'] = ofi_best
    return df


# ### Step 2

# In[8]:


def compute_multi_level_ofi(df, levels=10):
    multi_ofi = []
    prev = {f'bid_px_{i:02}': None for i in range(levels)}
    prev.update({f'ask_px_{i:02}': None for i in range(levels)})
    prev.update({f'bid_sz_{i:02}': None for i in range(levels)})
    prev.update({f'ask_sz_{i:02}': None for i in range(levels)})

    for _, row in df.iterrows():
        total_ofi = 0
        for i in range(levels):
            bpx = row[f'bid_px_{i:02}']
            bs = row[f'bid_sz_{i:02}']
            apx = row[f'ask_px_{i:02}']
            asp = row[f'ask_sz_{i:02}']

            if prev[f'bid_px_{i:02}'] is not None:
                total_ofi += (
                    bs if bpx > prev[f'bid_px_{i:02}'] else
                    (bs - prev[f'bid_sz_{i:02}'] if bpx == prev[f'bid_px_{i:02}'] else -prev[f'bid_sz_{i:02}'])
                )

            if prev[f'ask_px_{i:02}'] is not None:
                total_ofi += (
                    -asp if apx < prev[f'ask_px_{i:02}'] else
                    (prev[f'ask_sz_{i:02}'] - asp if apx == prev[f'ask_px_{i:02}'] else prev[f'ask_sz_{i:02}'])
                )

            prev[f'bid_px_{i:02}'], prev[f'bid_sz_{i:02}'] = bpx, bs
            prev[f'ask_px_{i:02}'], prev[f'ask_sz_{i:02}'] = apx, asp

        multi_ofi.append(total_ofi)

    df['OFI_multi'] = multi_ofi
    return df


# ### Step 3

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# 
# def compute_integrated_ofi(df, levels=10):
#     ofi_levels = []
#     for i in range(levels):
#         ofi = df[f'bid_sz_{i:02}'] - df[f'ask_sz_{i:02}']
#         ofi_levels.append(ofi)
# 
#     ofi_matrix = np.array(ofi_levels).T
#     scaled = StandardScaler().fit_transform(ofi_matrix)
# 
#     pca = PCA(n_components=1)
#     integrated = pca.fit_transform(scaled)
# 
#     df['OFI_integrated'] = integrated.flatten()
#     return df
# 

# ### Step 4

# In[12]:


def compute_cross_asset_ofi(df):
    df['OFI_cross'] = df.groupby('ts_event')['OFI_best'].transform(lambda x: x.mean())
    return df


# ### Step 5

# In[14]:


data = compute_best_level_ofi(data)
data = compute_multi_level_ofi(data)
data = compute_integrated_ofi(data)
data = compute_cross_asset_ofi(data)


# ### Output

# In[38]:


data[['ts_event', 'OFI_best', 'OFI_multi', 'OFI_integrated', 'OFI_cross']].head(10)


# ### PLOT

# In[23]:


import matplotlib.pyplot as plt
import pandas as pd

# Ensure timestamp column is datetime
data['ts_event'] = pd.to_datetime(data['ts_event'])
data = data.sort_values('ts_event')

# Plot all OFI features
plt.figure(figsize=(15, 6))

plt.plot(data['ts_event'], data['OFI_best'], label='Best-Level OFI', alpha=0.8, linewidth=1)
plt.plot(data['ts_event'], data['OFI_multi'], label='Multi-Level OFI', alpha=0.6, linewidth=1)
plt.plot(data['ts_event'], data['OFI_integrated'], label='Integrated OFI (PCA)', alpha=0.8, linewidth=1)
plt.plot(data['ts_event'], data['OFI_cross'], label='Cross-Asset OFI', alpha=0.8, linewidth=1)

plt.title("Order Flow Imbalance (OFI) Features Over Time")
plt.xlabel("Timestamp")
plt.ylabel("OFI Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ---
