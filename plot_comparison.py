import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

sb.set(font_scale=1.8)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
df = pd.read_csv("comparison.csv")

sb.boxplot(data=df, x="mode", y="time", ax=ax)
plt.ylabel("Runtime (Seconds)")
plt.xlabel("Implementation")
plt.savefig("comparison.png", bbox_inches="tight")
