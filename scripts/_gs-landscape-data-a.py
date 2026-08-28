import sys, time, numpy as np
sys.path.insert(0, sys.argv[1])
from symeval import Model

SP = sys.argv[1]
m = Model(f'{SP}/gs_sym.json')
CG = np.geomspace(4, 1 << 19, 900)          # cache sizes, in blocks

def mr_at(N, M):
    cs, mr, _ = m.mrc(p0=int(N), p1=int(M))
    return np.interp(CG, cs, mr)

def cmin(N, M, tau):
    v = mr_at(N, M)
    i = np.argmax(v <= tau)
    return CG[i] if v[i] <= tau else np.nan

t0 = time.time()
# panel a: MRC family, N fixed
Ms_a = [32, 64, 128, 256]
A = np.array([mr_at(128, M) for M in Ms_a])

# panel b: landscape over (C, M) at N = 128
Ms_b = np.unique(np.round(np.geomspace(16, 512, 44)).astype(int))
B = np.array([mr_at(128, M) for M in Ms_b])
print(f'panels a,b done {time.time()-t0:.0f}s', flush=True)

# panel c: min cache size for two targets, over (M, N)
Ns = np.unique(np.round(np.geomspace(16, 384, 22)).astype(int))
Msc = np.unique(np.round(np.geomspace(16, 384, 22)).astype(int))
C5 = np.zeros((len(Msc), len(Ns))); C1 = np.zeros_like(C5)
for i, M in enumerate(Msc):
    for j, N in enumerate(Ns):
        v = mr_at(N, M)
        for tau, out in ((0.05, C5), (0.01, C1)):
            k = np.argmax(v <= tau)
            out[i, j] = CG[k] if v[k] <= tau else np.nan
    print(f'  M={M} ({time.time()-t0:.0f}s)', flush=True)

np.savez(f'{SP}/figdata.npz', CG=CG, Ms_a=Ms_a, A=A, Ms_b=Ms_b, B=B,
         Ns=Ns, Msc=Msc, C5=C5, C1=C1)
print(f'total {time.time()-t0:.0f}s')
