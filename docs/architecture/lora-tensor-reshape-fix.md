# LoraTorchTensor.reshape() 修复文档

## 概述

修复 `convert_lora_to_gguf.py` 中 `LoraTorchTensor.reshape()` 对最后一维拆分的支持。
原代码在 `new_shape[-1] != orig_shape[-1]` 时直接 `raise NotImplementedError`，
导致 Qwen3.5 等模型的 LoRA adapter 转换失败（Issue #21125）。

## 问题根因

### LoRA 张量结构

`LoraTorchTensor` 将 LoRA 分解表示为两个独立张量：

```
A: (*extra_A, rank, row_size)     — 拥有行维度
B: (*leading_B, rank)             — 拥有列维度
有效形状: (*leading_B, *extra_A, row_size) = B@A 的输出形状
```

- B "拥有" 前导维度（对应原始张量的列维度）
- A "拥有" 尾部维度（对应原始张量的行维度）
- `rank` 是内部收缩维度

### 旧代码的局限

旧代码假设 `new_shape[-1] == orig_shape[-1]`（最后一维不变），即只有 B 的前导维度被拆分：

```python
if new_shape[-1] != orig_shape[-1]:
    raise NotImplementedError
```

当 HuggingFace 模型代码调用 `tensor.reshape()` 拆分最后一维时触发崩溃。

### 触发场景

`conversion/qwen.py` 中的 `_reorder_v_heads(tensor, dim=1)`：

```python
# 对于 out_proj weight: (out_dim, num_v_heads * head_dim)
# dim=1 → 拆分最后一维
new_shape = (out_dim, num_k_heads, num_v_per_k, head_dim)
tensor.reshape(*new_shape)  # new_shape[-1]=head_dim ≠ orig_shape[-1]=num_v_heads*head_dim
```

以及 HuggingFace 模型代码中通过 `torch.reshape()` 的任何类似调用（经 `__torch_function__` 拦截后进入 `LoraTorchTensor.reshape()`）。

## 修复方案

### 修改文件

`convert_lora_to_gguf.py`：`LoraTorchTensor` 类，+70/-10 行。

### 1. `__init__` — 放宽维度约束

```python
# 旧: assert len(A.shape) == len(B.shape)
# 新: 移除该断言，A/B 可有不同维度
assert A.shape[-2] == B.shape[-1]  # 只保留 rank 匹配检查
```

### 2. `shape` — 正确处理 A 的额外维度

```python
@property
def shape(self):
    n_diff = len(A.shape) - len(B.shape)
    if n_diff > 0:
        return (*B.shape[:-1], *A.shape[:n_diff], A.shape[-1])
    return (*B.shape[:-1], A.shape[-1])
```

当 A 有额外前导维度时（拆分最后一维的结果），将其插入 B 和 A_last 之间。

### 3. `reshape` — 核心修复

统一算法：

1. 通过 `prod(new_shape[:k]) == prod(B.shape[:-1])` 找到前导维度分界点 `k`
2. `shape_A = (*new_shape[k:-1], rank, new_shape[-1])`
3. `shape_B = (*new_shape[:k], rank)`
4. 如果 `n_old_extra != n_new_extra`，通过 **permute** 保证 rank 维度在 C-contiguous 布局中位置正确：

```
添加额外维度 (n_old < n_new):
  A: (rank, N) → reshape(rank, *extra, h) → permute → (*extra, rank, h)

移除额外维度 (n_old > n_new):
  A: (*extra, rank, h) → permute → (rank, *extra, h) → reshape(rank, N)
```

### 4. `permute` — 支持 A 有非 1 前导维度

原代码要求 Case 1 (`dims[-1] == -1`) 时 A 的前导维度全是 1：

```python
# 旧: assert all(dim == 1 for dim in A.shape[:-2])
# 新: 当 A 有非 1 前导维度时，同时 permute A 和 B
a_perm = [dst_eff - n_b_leading for each A extra dim]
b_perm = [convert_to_B_space(dims[i]) for each B leading dim]
```

## 测试验证

| 测试 | 描述 | 结果 |
|------|------|------|
| 测试1 | 拆分 B 前导维度（正常 reshape） | ✅ |
| 测试2 | 拆分第一维（B 吸收额外维度） | ✅ |
| 测试3 | 拆分最后一维（A 吸收额外维度） | ✅ |
| 测试4 | `_reorder_v_heads` dim=0 完整链条 | ✅ |
| 测试5 | `_reorder_v_heads` dim=1 完整链条 | ✅ |
| 测试6 | reshape 前后元素总数不变 | ✅ |
| 测试7 | B 维度多于 A 时 shape 正确 | ✅ |
| 全管道 | Q/K/V/O/FFN 层 reshape 模拟 | ✅ |
| 随机 | 100 次随机维度测试 | ✅ |

## 影响范围

- **只影响 LoRA adapter 转换**（`convert_lora_to_gguf.py`），不影响模型推理
- 向后兼容：所有原有测试通过
- 不改变 GGUF 输出格式，只影响转换过程中的内存重排

## 相关链接

- Issue: https://github.com/ggml-org/llama.cpp/issues/21125
- 相关函数: `conversion/qwen.py:_reorder_v_heads`, `_LinearAttentionVReorderBase.modify_tensors`
