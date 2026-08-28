"""Instantiate a fully-symbolic RI distribution (analyzer --json output) at
concrete parameter values and run the Denning recursion to get a miss-ratio
curve.  Mirrors analyzer/src/isl.rs::get_distro + denning::MissRatioCurve::new.
"""
import json, re, numpy as np, sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication)

TRANS = standard_transformations + (implicit_multiplication,)


def _defrac(s):
    """Rewrite \frac{A}{B} -> ((A)/(B)) with brace matching."""
    while True:
        i = s.find(r'\frac')
        if i < 0:
            return s
        j = i + len(r'\frac')
        while s[j] == ' ':
            j += 1
        parts = []
        for _ in range(2):
            assert s[j] == '{', s[j:j+20]
            depth, k = 0, j
            while True:
                if s[k] == '{':
                    depth += 1
                elif s[k] == '}':
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            parts.append(s[j+1:k])
            j = k + 1
        s = s[:i] + '((' + _defrac(parts[0]) + ')/(' + _defrac(parts[1]) + '))' + s[j:]


def latex_poly(s):
    s = _defrac(s)
    s = s.replace(r'\left(', '(').replace(r'\right)', ')')
    s = re.sub(r'\^\{([^}]*)\}', r'**(\1)', s)
    s = s.replace('{', '(').replace('}', ')')
    return parse_expr(s, transformations=TRANS)


CMP = re.compile(r'(<=|>=|<|>|=)')


def parse_range(s):
    """ISL-style guard -> list of sympy Boolean relations."""
    if not s.strip():
        return []
    out = []
    for atom in re.split(r'\s+and\s+', s.strip()):
        atom = atom.strip()
        m = re.match(r'^\((.*)\)\s*mod\s*(\d+)\s*=\s*(\d+)$', atom)
        if m:
            e = parse_expr(m.group(1), transformations=TRANS)
            out.append(sp.Eq(sp.Mod(e, int(m.group(2))), int(m.group(3))))
            continue
        toks = CMP.split(atom)
        exprs = [parse_expr(t, transformations=TRANS) for t in toks[0::2]]
        ops = toks[1::2]
        for k, op in enumerate(ops):
            a, b = exprs[k], exprs[k+1]
            out.append({'<': a < b, '<=': a <= b, '>': a > b,
                        '>=': a >= b, '=': sp.Eq(a, b)}[op])
    return out


class Model:
    """A derived symbolic RI distribution, evaluable at any parameter point."""

    def __init__(self, path):
        d = json.load(open(path))
        self.total = latex_poly(re.sub(r'^\\left\[.*?\\right\]\s*\\Rightarrow\s*',
                                       '', d['total_count']))
        self.items = []
        for ri, rng, cnt in zip(d['ri_values'], d['symbol_ranges'], d['counts']):
            self.items.append((latex_poly(ri), parse_range(rng), latex_poly(cnt)))
        self.params = sorted(self.total.free_symbols, key=lambda s: s.name)

    def ri_histogram(self, subs, box=None):
        """Enumerate integer points of every piece; return (ri, count) arrays."""
        subs = {sp.Symbol(k): v for k, v in subs.items()}
        pmax = max(subs.values()) if subs else 0
        lo, hi = (-8, pmax + 8) if box is None else box
        acc = {}
        for ri, rng, cnt in self.items:
            rng = [c.subs(subs) for c in rng]
            if any(c is sp.false for c in rng):
                continue
            rng = [c for c in rng if c is not sp.true]
            ri_s, cnt_s = ri.subs(subs), cnt.subs(subs)
            free = sorted(set().union(*[c.free_symbols for c in rng]) if rng else set(),
                          key=lambda s: s.name)
            free = [v for v in free
                    if v in ri_s.free_symbols or v in cnt_s.free_symbols or True]
            if not free:
                if rng and not all(bool(c) for c in rng):
                    continue
                _add(acc, float(ri_s), float(cnt_s))
                continue
            grids = np.meshgrid(*[np.arange(lo, hi + 1) for _ in free], indexing='ij')
            mask = np.ones(grids[0].shape, dtype=bool)
            for c in rng:
                mask &= _mask(c, free, grids)
            if not mask.any():
                continue
            pts = [g[mask] for g in grids]
            vals = _num(ri_s, free, pts)
            cnts = _num(cnt_s, free, pts)
            for v, c in zip(np.rint(vals).astype(np.int64), cnts):
                _add(acc, int(v), float(c))
        ks = np.array(sorted(acc), dtype=np.float64)
        vs = np.array([acc[int(k)] if float(k).is_integer() else acc[k] for k in ks])
        return ks, vs

    def mrc(self, box=None, **subs):
        """Denning recursion -> (cache sizes in blocks, miss ratios)."""
        ri, cnt = self.ri_histogram(subs, box=box)
        total = float(self.total.subs({sp.Symbol(k): v for k, v in subs.items()}))
        ri = np.concatenate([[0.0], ri])
        p = np.concatenate([[0.0], cnt]) / total
        mr = 1.0 - np.cumsum(p)                    # P(RI > ri_i)
        cs = np.concatenate([[0.0], np.cumsum(mr[:-1] * np.diff(ri))])
        return cs, np.clip(mr, 0.0, 1.0), total


def _add(acc, k, v):
    acc[k] = acc.get(k, 0.0) + v


def _num(expr, free, pts):
    if not expr.free_symbols:
        return np.full(pts[0].shape, float(expr))
    f = sp.lambdify(free, expr, 'numpy')
    return np.broadcast_to(np.asarray(f(*pts), dtype=np.float64), pts[0].shape)


def _mask(c, free, grids):
    f = sp.lambdify(free, c, [{'Mod': lambda a, b: np.mod(a, b)}, 'numpy'])
    r = f(*grids)
    return np.broadcast_to(np.asarray(r, dtype=bool), grids[0].shape)
