# import matplotlib.pyplot as plt
import scipy.stats as st
# import matplotlib
# matplotlib.use("agg")
# import seaborn as sns

#使用scipy生成具有偏态分布的loggamma分布的数据
x = st.loggamma.rvs(5, size=500) + 5
xt,_ = st.boxcox(x)
print(st.boxcox(x))
# #boxcox会同时返回两个参数，转化后的数据和最合适的λ值，此示例不需要返回λ值，因而设置为空
# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# sns.distplot(x,rug=True)
# ax2 = fig.add_subplot(212)
# sns.distplot(xt,rug=True)
# plt.savefig("test22.png",dpi=300)
# plt.close()