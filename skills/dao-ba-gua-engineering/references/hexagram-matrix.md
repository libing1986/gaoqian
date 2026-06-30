# 六十四卦交互矩阵 — 完整参考

## 原理

八卦的本质是八种基础属性（乾兑离震巽坎艮坤）。**八八六十四卦**就是把这八种属性两两组合，描述它们之间的交互关系。在工程中，这就是一个**8×8 的交互耦合矩阵**。

```
六十四卦 ∈ 八经卦²
每卦 = (下卦, 上卦) — "下卦提供，上卦消费"
```

## 第五层：马尔可夫收敛模型 — 动态预测

八卦的递归不仅是 8ⁿ 的组合展开，它本质上是一个**8 状态马尔可夫链**的幂级数演化。

### 数学本质

```
八卦 = 3 爻 × 2 值 = 2³ = 8 种状态 = 状态空间 S(|S|=8)
每层递归 = 转移矩阵 P[n] = Pⁿ
S(n) = 8¹ + 8² + 8³ + ... + 8ⁿ
        = P + P² + P³ + ... + Pⁿ
```

当 n→∞ 时，如果链是非周期不可约的，Pⁿ 的每一行收敛到同一个向量 π（平稳分布）：

```
π = π · P
```

这个 π 就是"最大公约数"——状态空间无限展开后最终的收敛态。

### P 矩阵的定义

```
P[i][j] = 改模块 i 后，模块 j 在被修改后的 N 次回归测试中
          出现故障的概率
         = count(改 i 后 j 红) / count(改 i 后跑回归)
```

### 完整 Python 实现

```python
import numpy as np
from typing import List, Dict, Tuple

class HexagramMarkovModel:
    """
    八卦马尔可夫链模型
    - 8 经卦 = 8 状态
    - P[i][j] = 改 i 影响 j 的概率
    - Pⁿ 收敛 → π（平稳分布）= "最大公约数"
    """
    
    MODULES = ['乾☰需求', '兑☱契约', '离☲架构', '震☳实现',
               '巽☴依赖', '坎☵风险', '艮☶回归', '坤☷部署']
    
    def __init__(self):
        self.P = np.full((8, 8), 0.125)  # 初始均匀分布
        self.history: List[Tuple[int, int, bool]] = []  # (i, j, failed)
        self.counts = np.zeros((8, 8))    # 改 i 后 j 的总回归次数
        self.fails = np.zeros((8, 8))     # 改 i 后 j 的失败次数
        self.pi = None
    
    def record(self, mod_i: int, mod_j: int, test_failed: bool):
        """记录一次回归测试结果"""
        self.counts[mod_i][mod_j] += 1
        if test_failed:
            self.fails[mod_i][mod_j] += 1
        self.history.append((mod_i, mod_j, test_failed))
        # 更新 P 矩阵（指数滑动平均）
        alpha = 0.7  # 新数据权重
        old = self.P[mod_i][mod_j]
        rate = self.fails[mod_i][mod_j] / max(self.counts[mod_i][mod_j], 1)
        self.P[mod_i][mod_j] = alpha * rate + (1 - alpha) * old
        # 重新归一化行
        self.P[mod_i] /= self.P[mod_i].sum()
    
    def compute_steady_state(self) -> np.ndarray:
        """解 π = π·P → 最大特征值的左特征向量"""
        vals, vecs = np.linalg.eig(self.P.T)
        pi = vecs[:, np.argmax(vals)]
        self.pi = pi.real / pi.real.sum()
        return self.pi
    
    def compute_P2(self) -> np.ndarray:
        """P² = 间接传播矩阵"""
        return self.P @ self.P
    
    def indirect_gap(self) -> np.ndarray:
        """P² - P = 间接耦合（直接看不出的依赖）"""
        return self.compute_P2() - self.P
    
    def propagate(self, start: int, steps: int = 50) -> np.ndarray:
        """从 start 出故障开始，n 步后的故障概率分布"""
        state = np.zeros(8)
        state[start] = 1.0
        for _ in range(steps):
            state = state @ self.P
        return state
    
    def top_vulnerable(self, n: int = 3) -> List[Tuple[str, float]]:
        """返回最脆弱的 n 个模块（稳态概率最大）"""
        if self.pi is None:
            self.compute_steady_state()
        idx = np.argsort(self.pi)[::-1]
        return [(self.MODULES[i], self.pi[i]) for i in idx[:n]]
    
    def top_influencers(self, n: int = 3) -> List[Tuple[str, float]]:
        """返回影响力最大的 n 个模块（行和最大）"""
        row_sums = self.P.sum(axis=1)  # 行和 = 对全局的影响总和
        idx = np.argsort(row_sums)[::-1]
        return [(self.MODULES[i], row_sums[i]) for i in idx[:n]]
    
    def intervention_effect(self, mod_i: int, reduction: float = 0.5) -> float:
        """
        如果降低模块 i 的影响概率 reduction%，新的 π 会变化多少？
        用于量化"这次重构到底降低了多少系统熵"
        """
        P_new = self.P.copy()
        for j in range(8):
            if j != mod_i:
                P_new[mod_i][j] *= (1 - reduction)
        P_new[mod_i][mod_i] = 1 - P_new[mod_i].sum() + P_new[mod_i][mod_i]
        vals, vecs = np.linalg.eig(P_new.T)
        pi_new = vecs[:, np.argmax(vals)].real
        pi_new = pi_new / pi_new.sum()
        delta = np.abs(pi_new - self.pi).sum()
        return delta


# ─── 使用示例 ───

model = HexagramMarkovModel()

# 模拟 1000 次部署记录
import random
for _ in range(1000):
    i = random.randint(0, 7)  # 改了哪个模块
    j = random.randint(0, 7)  # 回归测试哪个模块
    failed = random.random() < model.P[i][j]  # 按当前概率失败
    model.record(i, j, failed)

# 计算稳态
pi = model.compute_steady_state()
print("稳态分布 π (最大公约数):")
for name, prob in zip(model.MODULES, pi):
    print(f"  {name}: {prob:.3f}")

# 最脆弱的模块
print("\n最脆弱模块:")
for name, prob in model.top_vulnerable(3):
    print(f"  {name} (π={prob:.3f})")

# 间接耦合探测
gap = model.indirect_gap()
print("\n间接耦合（P²-P 中最大的 5 个）:")
indices = np.argsort(gap.flatten())[::-1][:5]
for idx in indices:
    i, j = divmod(idx, 8)
    if gap[i][j] > 0.05:
        print(f"  {model.MODULES[i]} →→ {model.MODULES[j]}: P²-P={gap[i][j]:.3f}")

# 干预效果
print("\n如果降低 震☳实现 的影响 50%:")
delta = model.intervention_effect(3, 0.5)
print(f"  系统平稳分布变化量: {delta:.4f}")
```

## 八阶段标准矩阵

| ↓下卦 → 上卦 | 乾☰ 需求 | 兑☱ 契约 | 离☲ 架构 | 震☳ 实现 | 巽☴ 依赖 | 坎☵ 风险 | 艮☶ 回归 | 坤☷ 部署 |
|---|---|---|---|---|---|---|---|---|
| **乾☰ 需求** | ─ | 高 | 高 | 中 | 低 | 低 | 低 | 中 |
| **兑☱ 契约** | — | ─ | 中 | 高 | 中 | 低 | 低 | 低 |
| **离☲ 架构** | — | — | ─ | 高 | 高 | 中 | 低 | 中 |
| **震☳ 实现** | — | — | — | ─ | 高 | 中 | 低 | 中 |
| **巽☴ 依赖** | — | — | — | — | ─ | 高 | 中 | 中 |
| **坎☵ 风险** | — | — | — | — | — | ─ | 强(门禁) | 低 |
| **艮☶ 回归** | — | — | — | — | — | — | ─ | 高(门禁) |
| **坤☷ 部署** | — | — | — | — | — | — | — | ─ |

### 耦合强度定义

| 强度 | 含义 | 示例 |
|------|------|------|
| **高** | 直接强依赖，改了 A 必须改 B | 架构改了 → 实现必须重写 |
| **中** | 间接依赖，改了 A 大概率影响 B | 需求变了 → 实现可能需要调整 |
| **低** | 弱关联，改了 A 可能需要关注 B | 需求变了 → 依赖配置可能变化 |
| **强(门禁)** | 硬性阻断，B 不通过 A 不能前进 | 回归没过 → 不能部署 |
| **─** | 自反/无关系 | 自己不影响自己 |

## 工具级六十四卦矩阵（创建模板）

### 快速构建方法

1. 列出你智能体中的所有活跃工具：`web_search`, `fetch_page`, `write_file`, `execute_code`, `analyze_image`, `call_api` 等
2. 创建一个 N×N 表格，N=工具数量
3. 对每个 (下卦工具, 上卦工具)，问：改了下卦，上卦会受影响吗？
4. 填耦合强度

### 示例模板

```python
# 六十四卦矩阵生成器
def build_hexagram_matrix(tools: list[str], dependencies: dict[tuple[str,str], str]) -> str:
    """
    tools: 模块/工具名列表
    dependencies: {(提供方, 消费方): 强度}
    
    用法:
    tools = ['web_search', 'fetch_page', 'write_file', 'sandbox_execute', 'analyze_image']
    deps = {
        ('web_search', 'fetch_page'): '高',  # 搜索提供了url给fetch
        ('web_search', 'write_file'): '中',  # 搜索结果可写入
        ('fetch_page', 'write_file'): '高',  # 抓取的内容写入
        ('sandbox_execute', 'write_file'): '中',
        ('sandbox_execute', 'fetch_page'): '低',
        ('analyze_image', 'write_file'): '中',
    }
    print(build_hexagram_matrix(tools, deps))
    """
    header = "| ↓提供方 → 消费方 | " + " | ".join(tools) + " |"
    sep = "|" + "---|" * (len(tools) + 1)
    lines = [header, sep]
    for t in tools:
        row = [f"**{t}**"]
        for u in tools:
            if t == u:
                row.append("─")
            else:
                strength = dependencies.get((t, u), "低")
                row.append(strength)
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
```

### 圈复杂度计算公式

```python
def hexagram_metrics(matrix: dict[str, dict[str, str]]):
    """
    计算每个模块的影响半径和被影响负债
    
    matrix: {提供方: {消费方: 强度}}
    
    返回: {模块: (影响半径, 被影响负债, 危险等级)}
    """
    modules = list(matrix.keys())
    weights = {'高': 3, '强': 3, '中': 2, '低': 1, '门禁': 3}
    
    result = {}
    for m in modules:
        # 影响半径 = 行总和（改了它影响谁）
        impact = sum(weights.get(matrix[m].get(n, '低'), 0) 
                    for n in modules if n != m)
        # 被影响负债 = 列总和（谁改了影响它）
        liability = sum(weights.get(matrix[n].get(m, '低'), 0)
                       for n in modules if n != m)
        
        if impact >= 6:
            danger = "⛔ 最危险"
        elif impact >= 4:
            danger = "⚠️ 高影响"
        elif impact >= 2:
            danger = "📌 需注意"
        else:
            danger = "✅ 低风险"
        
        result[m] = (impact, liability, danger)
    
    return result
```

## 六十四卦诊断流程

### 修改前检查（读行法）

```
1. 找到要改的模块在矩阵中的行
2. 标出所有"高/强/中"的格
3. 这些格对应的上卦就是需要同步修改的模块
4. 如果是"门禁"→ 必须先通过门禁条件
```

### 故障排查（读列法）

```
1. 找到出故障的模块在矩阵中的列
2. 标出所有"高/强/中"的格
3. 这些格对应的下卦就是优先排查方向
4. 问：最近谁改过这些模块？
```

## 自建矩阵练习

对活跃工具建立矩阵：

| ↓工具 → 消费方 | web_search | fetch_page | write_file | analyze_image | sandbox_execute |
|---|---|---|---|---|---|
| **web_search** | ─ | 高 | 中 | — | — |
| **fetch_page** | — | ─ | 高 | — | 中 |
| **write_file** | — | — | ─ | — | — |
| **analyze_image** | — | — | 中 | ─ | — |
| **sandbox_execute** | — | — | 中 | — | ─ |

圈复杂度计算：
- `web_search`: 影响半径=高+中=2+1=3 ✅ 低风险
- `fetch_page`: 影响半径=高+中=2+1=3 ✅ 低风险  
- `write_file`: 影响半径=0 ⚠️ **极高被影响负债**（被4个下游消费）
- `analyze_image`: 影响半径=中=1 ✅
- `sandbox_execute`: 影响半径=中=1 ✅
