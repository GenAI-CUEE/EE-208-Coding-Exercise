"""
Reference solution / answer key for matmul_flops_exercise.md
Run this to verify the exercise is well-posed and to get real numbers
for the expected benchmark table.
"""
import numpy as np
import time


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Inner dimensions must match"

    C = np.zeros((M, N))
    # TODO: implement with three nested for-loops
    # for i in range(M):
    #     for j in range(N):
    #         for k in range(K):
    #   
    return C


def flops_matmul(M, K, N):
    return 2 * M * K * N


def time_it(fn, A, B, repeats=1):
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(A, B)
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed, result


if __name__ == "__main__":
    # --- Correctness check ---
    A = np.random.rand(10, 12)
    B = np.random.rand(12, 8)
    assert np.allclose(matmul_naive(A, B), A @ B)
    print("Correctness check passed.\n")

    # --- Benchmark ---
    sizes = [64, 128, 256, 512]
    print(f"{'n':>4} {'FLOPs':>14} {'t_naive(s)':>11} {'GF/s naive':>11} "
          f"{'t_numpy(s)':>11} {'GF/s numpy':>11} {'speedup':>9}")

    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        f = flops_matmul(n, n, n)

        t_naive, _ = time_it(matmul_naive, A, B, repeats=1)
        t_numpy, _ = time_it(lambda X, Y: X @ Y, A, B, repeats=5)

        gflops_naive = f / (t_naive * 1e9)
        gflops_numpy = f / (t_numpy * 1e9)
        speedup = t_naive / t_numpy

        print(f"{n:4d} {f:14,d} {t_naive:11.4f} {gflops_naive:11.3f} "
              f"{t_numpy:11.6f} {gflops_numpy:11.2f} {speedup:9.1f}x")
