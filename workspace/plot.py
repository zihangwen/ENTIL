# %%
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np

# %%
spinup_data = pd.read_csv('/home/zihangw/ENTIL/data/ppo_HalfCheetah_0/ppo_HalfCheetah_s0/progress.txt', sep='\t')

with open('/home/zihangw/ENTIL/data/ENTIL/HalfCheetah-v5/logger.pkl', 'rb') as f:
    entil_logger = pickle.load(f)

df_entil = pd.DataFrame(entil_logger)

# %%

plt.plot(
    (df_entil["EpLen"] * df_entil["NEnv"]).cumsum(),
    df_entil["AverageEpRet"],
    label='ENTIL (PPO only)'
)
plt.fill_between(
    (df_entil["EpLen"] * df_entil["NEnv"]).cumsum(),
    df_entil["AverageEpRet"] - df_entil["StdEpRet"],
    df_entil["AverageEpRet"] + df_entil["StdEpRet"],
    alpha=0.2
)

plt.plot(
    spinup_data['TotalEnvInteracts'],
    spinup_data['AverageEpRet'],
    label='Spinup'
)
plt.fill_between(
    spinup_data['TotalEnvInteracts'],
    spinup_data['AverageEpRet'] - spinup_data['StdEpRet'],
    spinup_data['AverageEpRet'] + spinup_data['StdEpRet'],
    alpha=0.2
)

plt.legend()
plt.xlabel('TotalEnvInteracts')
plt.ylabel('AverageEpRet')
plt.title('HalfCheetah-v5')

# %%
