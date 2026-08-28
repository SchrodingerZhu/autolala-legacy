import sys, time, numpy as np
sys.path.insert(0, sys.argv[1]); SP = sys.argv[1]
from symeval import Model
m = Model(f'{SP}/gs_sym.json')
CG = np.geomspace(4, 1 << 19, 900)
def mr_at(N, M):
    cs, mr, _ = m.mrc(p0=int(N), p1=int(M)); return np.interp(CG, cs, mr)
Ns  = np.unique(np.round(np.geomspace(16, 384, 26)).astype(int))
Ms  = np.unique(np.round(np.geomspace(16, 384, 26)).astype(int))
cube = np.zeros((len(Ms), len(Ns), len(CG)))
t0 = time.time()
for i, M in enumerate(Ms):
    for j, N in enumerate(Ns):
        cube[i, j] = mr_at(N, M)
    print(f'M={M} {time.time()-t0:.0f}s', flush=True)
np.savez_compressed(f'{SP}/cube.npz', CG=CG, Ns=Ns, Ms=Ms, cube=cube)
print('done', time.time()-t0)
