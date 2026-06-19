# Moving Average and FLOP Count

**Course:** 2102208 — Programming for Electrical Engineering

This programming exercise asks you to implement a sliding-window **moving average** over a list of integers, and to track the **FLOP (floating-point operation) count** incurred while computing it.

---

## Problem

Given a list of `n` integers `a = [a1, a2, ..., an]` and an odd window size `k`, compute a moving average `o = [o1, o2, ..., on]` defined as:

$$
o_i =
\begin{cases}
\dfrac{1}{k} \displaystyle\sum_{j \in [\,i-\lfloor k/2 \rfloor \,:\, i+\lfloor k/2 \rfloor\,]} a_j, & \text{if } \lfloor k/2 \rfloor \leq i \leq n - \lfloor k/2 \rfloor \\[1em]
0, & \text{otherwise}
\end{cases}
$$

for $i = 1, 2, \dots, n$, where $\lfloor \cdot \rfloor$ is the floor operation, and indices are **1-based**.

In words: for each position `i`, average the `k` elements centered on `i`. If the window of size `k` would run off either end of the list, output `0` for that position instead.

---

## Input Format

| Input | Description |
|---|---|
| `a` | A list of `n` integers: `[a1, a2, ..., an]` |
| `k` | An odd integer — the window size |

## Output Format

| Output | Description |
|---|---|
| `output` | A list of `n` floats: `[o1, o2, ..., on]` |
| `flop` | An integer — the total FLOP count of the moving-average computation |

---

## Constraints

- `3 ≤ n ≤ 100`
- `k` is odd, and `k ≤ n - 2`
- `1 ≤ ai ≤ 100` for all `i`

---

## Sample

**Input:**

```python
a = [1, 2, 3, 4, 5, 6]
k = 3
```

**Output:**

```python
[0, 2.0, 3.0, 4.0, 5.0, 0], 12
```

### Worked example

With `k = 3`, `⌊k/2⌋ = 1`, so valid positions are `1 ≤ i ≤ n - 1`, i.e. `i = 2, 3, 4, 5` (1-indexed). Positions `i = 1` and `i = 6` fall outside the valid range and are `0`.

| `i` | Window (`a[i-1], a[i], a[i+1]`) | Sum | Average |
|---|---|---|---|
| 1 | — (window runs off the left edge) | — | `0` |
| 2 | `1, 2, 3` | 6 | `2.0` |
| 3 | `2, 3, 4` | 9 | `3.0` |
| 4 | `3, 4, 5` | 12 | `4.0` |
| 5 | `4, 5, 6` | 15 | `5.0` |
| 6 | — (window runs off the right edge) | — | `0` |

---

## Implementation

Fill in the following function:

```python
def moving_average(list_a: list, k: int):
    ...
    return output, flop  # Return the resulting list and the number of flop counts.
```

### ⚠️ Warning

**You are NOT ALLOWED to use the NumPy package.** This exercise is meant to be solved with plain Python (loops, lists, arithmetic) so that you compute — and can verify — the FLOP count by hand, not via a vectorized library call.

---

## Counting FLOPs

A FLOP (floating-point operation) here means one arithmetic operation: an addition, subtraction, multiplication, or division.

For each **valid** output position (where the window fully fits inside the list):

- Summing `k` elements takes `k - 1` additions.
- Dividing the sum by `k` takes `1` division.
- **Total per valid position:** `k - 1 + 1 = k` FLOPs.

For each **invalid** position (window runs off either edge), the output is simply `0` — no arithmetic is performed, so it contributes **0 FLOPs**.

**Total FLOP count:**

$$
\text{flop} = (\text{number of valid positions}) \times k
$$

### Checking the sample

- `n = 6`, `k = 3` → valid positions are `i = 2, 3, 4, 5`, i.e. **4** valid positions.
- `flop = 4 × k = 4 × 3 = 12` ✅ — matches the sample output.

---

## Tips

- Be careful with **1-based vs. 0-based indexing** when translating the formula into Python (Python lists are 0-indexed; the problem statement uses 1-indexed `i`).
- Compute `half = k // 2` once (equivalent to `⌊k/2⌋`), and use it to determine both the valid index range and the window slice bounds.
- Increment your `flop` counter only inside the branch where you actually do arithmetic — don't count FLOPs for positions that are just set to `0`.
- Double-check your valid-index range against the constraint `k ≤ n - 2`, which guarantees at least one valid output position exists.

See `main.pdf` for the full exercise explaination in 
<p align="center">
  <img src="PPT.png" alt="PPT" width="500">
</p>
