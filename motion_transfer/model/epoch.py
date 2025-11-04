import torch, time

x = torch.rand((128, 128, 128), device="cuda")
t0 = time.time()
for _ in range(1000):
    y = torch.matmul(x, x)
torch.cuda.synchronize()
print("GPU test time:", time.time() - t0)
