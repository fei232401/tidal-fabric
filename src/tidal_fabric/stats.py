"""统计工具:分位数(线性插值)与均值。确定性:纯函数,无随机。"""


def percentile(values, p):
    """p ∈ (0, 100];线性插值分位。空序列返回 None。"""
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def mean(values):
    return sum(values) / len(values) if values else None
