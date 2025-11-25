import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 参数
a = 7.85
beta_min = 0.0
beta_max = 0.005

# p 从 0 到 1
p = np.linspace(0.0, 1.0, 501)

# 原始 sigmoid，以 0.5 为中心
f_raw = 1 / (1 + np.exp(-a * (p - 0.5)))

# 两端值（用来做归一化）
f0 = 1 / (1 + np.exp(-a * (0.0 - 0.5)))
f1 = 1 / (1 + np.exp(-a * (1.0 - 0.5)))

# 归一化到 [0, 1]
f = (f_raw - f0) / (f1 - f0)

# 映射到 [beta_min, beta_max]
beta = beta_min + f * (beta_max - beta_min)

# ---- 用 seaborn 画图 ----
sns.set(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.lineplot(x=p, y=beta, label=r"$\beta(n_{\mathrm{eff}})$")

plt.xlabel(r"$n_{\mathrm{eff}}$")
plt.ylabel(r"$\beta$")
plt.tight_layout()
plt.show()
