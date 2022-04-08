import math

import numba
import numpy as np


def lpw_filter(f, dt, m, is_neg_allowed=True):
    D = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
    fact = 2 * f * dt
    arg = fact * math.pi
    lpw = [fact]

    for i in range(1, m+1):
        lpw.append(math.sin(arg * i) / (math.pi * i))

    lpw[m] /= 2.0

    # P130
    sumg = lpw[0]
    for i in range(1, m+1):
        sum = D[0]
        arg = math.pi * i / m
        for j in range(1, 4):
            sum += 2.0 * D[j] * math.cos(arg * j)
        lpw[i] *= sum
        sumg += 2*lpw[i]

    for i in range(1, m+1):
        lpw[i] /= sumg

    x_s = []
    x = 0
    for i in range(len(lpw)):
        x_s.append(x)
        x += 1

    if not is_neg_allowed:
        return [x_s, lpw]

    lpw_flipped = []
    x_s2 = []
    for i in range(0, len(lpw)):
        lpw_flipped.append(lpw[len(lpw) - 1 - i])
        x_s2.append(x_s[len(lpw) - 1 - i] * -1.0)

    return [x_s2[:len(lpw) - 1] + x_s,
            lpw_flipped[:len(lpw) - 1] + lpw]


def hpw_filter(f, dt, m):
    values = lpw_filter(f, dt, m)[1]

    for i in range(len(values)):
        if i == m:
            values[i] = 1.0 - values[i]
        else:
            values[i] = -values[i]

    return values


def bpw_filter(f1, f2, dt, m):
    val1 = lpw_filter(f1, dt, m)
    val2 = lpw_filter(f2, dt, m)

    for i in range(len(val1[0])):
        val1[1][i] = val2[1][i] - val1[1][i]

    return val1


def bsw_filter(f1, f2, dt, m):
    val1 = lpw_filter(f1, dt, m)
    val2 = lpw_filter(f2, dt, m)

    for i in range(len(val1[1])):
        if i == m:
            val1[1][i] = 1.0 + val1[1][i] - val2[1][i]
        else:
            val1[1][i] = val1[1][i] - val2[1][i]

    return val1
