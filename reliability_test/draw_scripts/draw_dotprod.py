# -*- coding: utf-8 -*-
import copy
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------- NTT Function -------------------
def ntt(a, mod, root):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        wlen = pow(root, (mod - 1) // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w % mod
                a[i + j] = (u + v) % mod
                a[i + j + length // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        length *= 2
    return a

# ------------------- NTT Data -------------------
n = 1024
mod = 15728641
root = 3
a = [(i % 16) for i in range(n)]
A = ntt(copy.deepcopy(a), mod, root)
a_noisy = copy.deepcopy(a)
a_noisy[10] = (a_noisy[10] + 1) % mod
A_noisy = ntt(copy.deepcopy(a_noisy), mod, root)

# ------------------- ResNet Data -------------------
data_resnet = [(1024, 6.0),
 (1024, 8.173205382760385),
 (1024, 8.241812291278775),
 (1024, 8.178290525685737),
 (1024, 8.158198897845523),
 (2048, 6.0397569563748705),
 (2048, 9.381722622648224),
 (2048, 9.23461208090798),
 (2048, 9.385137292480938),
 (2048, 9.396168134925254),
 (4096, 13.633131643885411),
 (4096, 9.336244480672157),
 (4096, 9.300053704326817),
 (4096, 9.372846794813823),
 (4096, 9.3397493297435),
 (8192, 9.233939740348413),
 (8192, 9.317223527520952),
 (8192, 13.537669900629016),
 (8192, 9.239931536939807),
 (8192, 9.220735015190797),
 (16384, 9.218962015475006),
 (16384, 9.215214796148706),
 (16384, 13.553095110606655),
 (16384, 9.235688101556606),
 (16384, 9.272455104891375),
 (32768, 9.284112797659601),
 (32768, 11.403703831503933),
 (32768, 9.19486734471826),
 (32768, 9.268625579043231),
 (32768, 9.262356889928643),
 (65536, 9.302410868281786),
 (65536, 9.242572925581769),
 (65536, 9.28547036796006),
 (65536, 6.039402629751354),
 (65536, 9.297098010249458),
 (131072, 12.507209365953798),
 (131072, 10.36062813721953),
 (131072, 15.75972061038529),
 (131072, 12.521557774050645),
 (131072, 12.520185091505606),
 (262144, 16.7668010199055),
 (262144, 16.792678157604207),
 (262144, 13.568607699001696),
 (262144, 16.78454413435171),
 (262144, 16.783295325697676),
 (524288, 35.10072653709297),
 (524288, 31.91172906594591),
 (524288, 35.21971613378339),
 (524288, 41.6728668391922),
 (524288, 35.1360925106),
 (1048576, 52.424675377399815),
 (1048576, 42.679093105818446),
 (1048576, 42.7594240041364),
 (1048576, 42.73659558471534),
 (1048576, 42.7605756302773),
 (2097152, 75.04558287804791),
 (2097152, 64.31507906420453),
 (2097152, 74.0439694362853),
 (2097152, 64.3248052673914),
 (2097152, 68.58413623876437),
 (4194304, 83.69646411145557),
 (4194304, 78.35989977665209),
 (4194304, 83.76668565469625),
 (4194304, 83.69128114238103),
 (4194304, 83.71790653877012),
 (8388608, 88.00094644428464),
 (8388608, 89.04579733569778),
 (8388608, 88.023311926955),
 (8388608, 88.06237314465922),
 (8388608, 87.98907065024139),
 (16777216, 90.139402276873),
 (16777216, 90.17931829891184),
 (16777216, 90.13217921180976),
 (16777216, 88.03936740387348),
 (16777216, 90.19131646851363),
 (33554432, 90.80636793421442),
 (33554432, 90.86115699667268),
 (33554432, 90.83155168264467),
 (33554432, 90.82055466981232),
 (33554432, 93.0),
 (67108864, 91.45101884185313),
 (67108864, 91.43034076526422),
 (67108864, 91.45285944009598),
 (67108864, 91.43915101786911),
 (67108864, 92.49722069580962),
 (134217728, 91.67477690214396),
 (134217728, 91.6663568960546),
 (134217728, 91.6699269670602),
 (134217728, 91.66114463180521),
 (134217728, 92.75812401974326)]

data_resnet_baseline =[(1024, 85.64229944436931),
 (1024, 85.10989187085029),
 (1024, 85.0),
 (1024, 85.0),
 (1024, 85.58774304654833),
 (2048, 86.65038050660951),
 (2048, 85.51584835399197),
 (2048, 85.74442598217964),
 (2048, 85.78338720274064),
 (2048, 85.71078025180078),
 (4096, 86.97038623423002),
 (4096, 86.18277185612912),
 (4096, 85.93233992331842),
 (4096, 87.40234279380698),
 (4096, 87.53500674599773),
 (8192, 86.91225462920487),
 (8192, 87.84420785444729),
 (8192, 87.41869146811693),
 (8192, 86.82679754522901),
 (8192, 86.59302115255008),
 (16384, 88.53044811588563),
 (16384, 87.89076766705863),
 (16384, 87.82228950725595),
 (16384, 87.10996417967029),
 (16384, 87.15151518472207),
 (32768, 88.58303352365107),
 (32768, 87.27354106570274),
 (32768, 87.4447883009034),
 (32768, 88.24207936606265),
 (32768, 87.8288677130644),
 (65536, 88.20213043555698),
 (65536, 88.73395440364882),
 (65536, 88.1626773975607),
 (65536, 88.34918800864908),
 (65536, 88.10878457822069),
 (131072, 88.92024214893992),
 (131072, 89.09402239015151),
 (131072, 88.36048535759295),
 (131072, 87.59792123533313),
 (131072, 87.59233901846451),
 (262144, 87.96747679164963),
 (262144, 87.8322418176358),
 (262144, 87.90857278295572),
 (262144, 88.27248581143425),
 (262144, 87.9189284842322),
 (524288, 87.72810804765686),
 (524288, 89.22972022930111),
 (524288, 88.01844446864145),
 (524288, 88.69964741529492),
 (524288, 88.65633496097259),
 (1048576, 90.08516836326342),
 (1048576, 91.36430035592468),
 (1048576, 89.97581953927212),
 (1048576, 91.1195863199127),
 (1048576, 90.49648718764557),
 (2097152, 91.35647225735667),
 (2097152, 91.95496303270455),
 (2097152, 91.56840221693),
 (2097152, 91.57309872485752),
 (2097152, 92.28488647320471),
 (4194304, 92.14970550874749),
 (4194304, 91.18097981568607),
 (4194304, 90.94416491589296),
 (4194304, 91.244871667877),
 (4194304, 91.62954293610701),
 (8388608, 92.47766332302928),
 (8388608, 92.56780065684447),
 (8388608, 92.47991508608008),
 (8388608, 91.4665852811777),
 (8388608, 91.15014990075495),
 (16777216, 92.50566582196865),
 (16777216, 93.2772595702379),
 (16777216, 92.11210338081897),
 (16777216, 93.4),
 (16777216, 92.37718722697144),
 (33554432, 93.4),
 (33554432, 92.84705330526918),
 (33554432, 92.25516644159154),
 (33554432, 92.14563666978252),
 (33554432, 91.89061697077712),
 (67108864, 93.4),
 (67108864, 93.0023045595725),
 (67108864, 93.37251843758483),
 (67108864, 92.89171222708381),
 (67108864, 93.26187981467658),
 (134217728, 93.4),
 (134217728, 93.4),
 (134217728, 92.56377881619457),
 (134217728, 93.02291051790948),
 (134217728, 93.10131434199126)]

# ------------------- Plot Styling -------------------
plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20

# ------------------- Create Figure with Custom Layout -------------------
fig = plt.figure(figsize=(15, 4), dpi=300)
gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.1, 1.1], hspace=0.45)

# Left column: Two stacked NTT subplots (a)
ax0_top = fig.add_subplot(gs[0, 0])
ax0_bottom = fig.add_subplot(gs[1, 0])


# Middle column: ResNet accuracy plot (b)
ax1 = fig.add_subplot(gs[:, 1])

# Right column: Stacked bar chart (c)
ax2 = fig.add_subplot(gs[:, 2])

# ------------------- Plot (a) Top: NTT Input -------------------
ax0_top.plot(a[:128], '-', label='Original Input', color="dimgrey", lw=2)
# ax0_top.plot(a_noisy[:128], '--', label='Original Input', color="red", lw=1)
noisy_index = 10
if noisy_index < 128:
    ax0_top.scatter(noisy_index, a[noisy_index], color='red', s=30, zorder=5, marker='^')
# ax0_top.set_title("NTT Input", fontsize=14, pad=5)
# ax0_top.set_xlabel("Index", fontsize=12)
# ax0_top.set_ylabel("Value", fontsize=16)
ax0_top.tick_params(axis='both', labelsize=16)
ax0_top.grid(True, alpha=0.3)

# ------------------- Plot (a) Bottom: NTT Output -------------------
ax0_bottom.plot(A_noisy[:128], '-', label='Noisy NTT', color='orange')
ax0_bottom.plot(A[:128], '-', label='Original NTT', lw=2, color="blue")
# ax0_bottom.set_title("NTT Output", fontsize=14, pad=5)
ax0_bottom.set_xlabel("Index", fontsize=23)
# ax0_bottom.set_ylabel("Value", fontsize=23)
# ax0_bottom.legend(loc="upper left", fontsize=15,)
ax0_bottom.tick_params(axis='both', labelsize=23)
ax0_bottom.grid(True, alpha=0.3)

ax0_bottom.set_yticklabels([])
ax0_top.set_yticklabels([])
ax0_top.set_xticklabels([])
# ------------------- Plot (b): ResNet Accuracy -------------------
colors = ['red', 'navy']
legends = ['CKKS', 'Baseline']
for idx, data in enumerate([data_resnet, data_resnet_baseline]):
    probs_raw, accs_raw = zip(*data)
    inv_probs_raw = [1 / p for p in probs_raw]

    grouped = defaultdict(list)
    for prob, acc in data:
        grouped[prob].append(acc)

    probs_unique = sorted(grouped.keys())
    inv_probs_unique = [1 / p for p in probs_unique]
    means = [np.mean(grouped[p]) for p in probs_unique]
    stds = [np.std(grouped[p], ddof=0) for p in probs_unique]
    
    # ax1.errorbar(
    #     inv_probs_unique,
    #     means,
    #     yerr=stds,
    #     fmt="-o",
    #     color=colors[idx],
    #     label=legends[idx],
    #     ecolor="black",
    #     capsize=5,
    # )
    ax1.plot(inv_probs_unique, means, "-o", color=colors[idx], label=legends[idx])

    # 绘制阴影区域（均值 ± 标准差）
    ax1.fill_between(
        inv_probs_unique,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        color=colors[idx],
        alpha=0.2  # 透明度，可调
    )
    ax1.set_ylim(0, 100)

ax1.set_xlabel("Error Rate")
ax1.set_ylabel("Accuracy (%)")
ax1.set_xscale('log')
ax1.legend(loc='lower left', fontsize=16, frameon=False)
ax1.grid(True, alpha=0.3)

# ------------------- Plot (c): Stacked Bar Chart -------------------
dnum = [1, 2, 3, 4, 6, 8, 12, '24\n(max)']
NTT = [36.6, 42.6, 48.2, 58.6, 62.6, 69.2, 72.1, 73]
BaseConv = [55, 48, 42, 31.2, 26, 18, 14, 11.835]
Modmul = [7.3, 8.3, 8.7, 9.1, 10.3, 11.7, 12.8, 14.065]
Others = [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]

bar_width = 0.6
x = np.arange(len(dnum))

ax2.bar(x, NTT, width=bar_width, label='NTT', alpha=0.5, color="blue")
ax2.bar(x, BaseConv, width=bar_width, bottom=NTT, label='BaseConv', alpha=0.6,color="orange")
ax2.bar(x, Modmul, width=bar_width, bottom=np.array(NTT)+np.array(BaseConv), label='Modmul', alpha=0.8, color="red")
ax2.bar(x, Others, width=bar_width, bottom=np.array(NTT)+np.array(BaseConv)+np.array(Modmul), label='Others', color='black')

ax2.set_xticks(x)
ax2.set_xticklabels(dnum)
ax2.set_xlabel('dnum')
ax2.set_ylabel('Computational Complexity')
ax2.legend(loc="lower right", fontsize=16)

# ------------------- Add Labels (a), (b), (c) -------------------
# ax0_top.text(-0.12, 1.08, '(a)', transform=ax0_top.transAxes,
#              fontsize=25, va='top', ha='left')
# ax1.text(-0.12, 1.08, '(b)', transform=ax1.transAxes,
#          fontsize=25, va='top', ha='left')
# ax2.text(-0.12, 1.08, '(c)', transform=ax2.transAxes,
#          fontsize=25, va='top', ha='left')

plt.tight_layout()
plt.subplots_adjust(wspace=0.4, hspace=0.5, top=0.92)
plt.savefig("./figures/eva_0_motivation.jpg", pad_inches=0.1, bbox_inches='tight')
print("Figure saved to eva_0_motivation.jpg")
