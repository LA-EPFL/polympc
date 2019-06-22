import numpy as np
import matplotlib.pyplot as plt

dense = [
[15, 26.920000, 4.089988],
[21, 100.600000, 25.513933],
[30, 96.350000, 29.877321],
[60, 262.010000, 82.534162],
[90, 349.940000, 79.860637],
[150, 812.040000, 155.581336],
[210, 1569.060000, 328.161269],
[300, 3386.800000, 544.574218],
]
dense = np.array(dense);

sparse_update = [ # counting update instead of setup time
[15, 15.620000, 1.665113],
[21, 44.010000, 2.172441],
[30, 40.150000, 11.082306],
[60, 159.970000, 38.648797],
[90, 200.520000, 76.645938],
[150, 468.780000, 67.243763],
[210, 893.220000, 161.074056],
[300, 1701.640000, 210.471448],
]
sparse_update = np.array(sparse_update);
sparse = [
[15, 25.680000, 11.248493],
[21, 53.280000, 4.403635],
[30, 88.630000, 23.579294],
[60, 171.710000, 56.130978],
[90, 257.700000, 32.466262],
[150, 771.380000, 116.579136],
[210, 1393.160000, 160.123989],
[300, 2814.310000, 487.813828],
]
sparse = np.array(sparse);

dense[:,1:] /= 1e3
sparse[:,1:] /= 1e3
sparse_update[:,1:] /= 1e3

# plt.plot(dense[:,0], dense[:,1], marker='.', label='dense')
# plt.plot(sparse[:,0], sparse[:,1], marker='.', label='sparse')
# plt.plot(sparse_update[:,0], sparse_update[:,1], marker='.', label='sparse (update)')

kwargs = dict(
    marker='.',
    capsize=3,
    ms=7,
    ecolor='grey',
    elinewidth=1,
    # alpha=0.7,
    # linewidth=2
)

figsize = np.array([6.4, 4.8])
figsize = 0.7*figsize
fig, ax = plt.subplots(figsize=figsize)

plt.errorbar(dense[:,0], dense[:,1], dense[:,2], color='C0', label='Eigen dense LDLT', **kwargs)
plt.errorbar(sparse[:,0], sparse[:,1], sparse[:,2], color='C1', label='Eigen sparse LDLT', **kwargs)
plt.errorbar(sparse_update[:,0], sparse_update[:,1], sparse_update[:,2], color='C2', label='Eigen sparse LDLT (caching)', **kwargs)
plt.legend()
# plt.ylabel('solve time [$\mu s$]')
plt.ylabel('solve time [ms]')
plt.xlabel('problem size')
plt.savefig('random_qp.pdf', bbox_inches='tight')
plt.show()
