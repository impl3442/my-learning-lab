import time
import numpy as np
import torch

class MatrixLab:
    def __init__(self):
        self.device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.specs = [
            (3, 3, 3, "3x3"),
            (3, 4, 2, "3x4*4x2"),
            (100, 100, 100, "100x100")
        ]
        # 存儲結構：{dim_label: {env_name: time}}
        self.raw_data = {s[3]: {} for s in self.specs}

    def _timer(self, func, *args):
        start = time.perf_counter()
        func(*args)
        if "gpu" in func.__name__ and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter() - start

    # --- 四種環境實作 ---
    def run_python(self, m, n, p):
        A = [[1.0]*n for _ in range(m)]; B = [[1.0]*p for _ in range(n)]
        C = [[0.0]*p for _ in range(m)]
        for i in range(m):
            for j in range(p):
                for k in range(n): C[i][j] += A[i][k] * B[k][j]

    def run_numpy(self, m, n, p):
        return np.ones((m, n)) @ np.ones((n, p))

    def run_torch_cpu(self, m, n, p):
        return torch.mm(torch.ones(m, n), torch.ones(n, p))

    def run_torch_gpu(self, m, n, p):
        if not torch.cuda.is_available(): return None
        return torch.mm(torch.ones(m, n, device=self.device_gpu), 
                        torch.ones(n, p, device=self.device_gpu))

    def execute(self):
        envs = [("Python", self.run_python), ("NumPy", self.run_numpy), 
                ("PT_CPU", self.run_torch_cpu), ("PT_GPU", self.run_torch_gpu)]

        for m, n, p, label in self.specs:
            for name, func in envs:
                t = self._timer(func, m, n, p)
                self.raw_data[label][name] = t

    def report(self):
        print(f"\n[2026-04-19] Matrix Lab: 加速倍率實測 (基準: Python 原生 1x)")
        print(f"{'Dimension':<12} | {'Env':<10} | {'Time (s)':<12} | {'Speedup'}")
        print("-" * 55)
        
        for dim, results in self.raw_data.items():
            base_time = results["Python"]
            for env, t in results.items():
                speedup = base_time / t if t > 0 else 0
                print(f"{dim:<12} | {env:<10} | {t:.8f} | {speedup:>8.1f}x")
            print("-" * 55)

if __name__ == "__main__":
    lab = MatrixLab()
    lab.execute()
    lab.report()
