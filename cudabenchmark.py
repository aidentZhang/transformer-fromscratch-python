import cupy as cp
import time

N = 10000  # adjust size if needed
A = cp.random.randn(N, N, dtype=cp.float32)
B = cp.random.randn(N, N, dtype=cp.float32)

start = time.time()
C = cp.matmul(A, B)
cp.cuda.Device().synchronize()  # wait for GPU
end = time.time()

print("GPU matmul time:", end - start)
