"""Rabi-driven two-tone sideband preparation of bosonic states: optimize, ship, simulate.

Helper module for rabi_sideband_pipeline.ipynb. The three stages are

1. effective-Hamiltonian optimal control (TensorFlow) in the displaced, dressed frame,
       H_eff = -g_eff [e^{i phi1} s+ a^dag + e^{i phi2} s- a^dag + h.c.],   g_eff = alpha chi / 2,
   phi_j(t) = sum_k C_jk (cos w_k t - 1) + S_jk sin w_k t, with the smoothness term and the
   phase-derivative hinge (thesis Eqs. 6.1-6.6);
2. export: rescale along the alpha*T invariant, beat-aligned schedule on the 4 ns grid, and the
   solution-folder format read by make_two_sideband_amp_oc.py, make_rabi_oc_amp.py and
   oc_cell_from_params.py;
3. displaced-frame simulation of the complete sequence (QuTiP), with time-dependent or constant
   Stark compensation and a search over beat-aligned, grid-integer segment lengths.

Adapted from cat_charfunc_low_alpha.ipynb, gkp_charfunc_delta.ipynb and
paper_rabi_sideband/scripts/{multistart,gkp_strong_drive,stark_sim,approx_vs_alpha}.py.
The shipped folders were written by the same conversion and verification code.
"""
import itertools
import os
import time
from math import lgamma
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre

ROOT = Path(__file__).resolve().parent
TWO_PI = 2*np.pi

# Device and design numbers (thesis Tab. 8.1 and Ch. 6). Every shipped design uses the nominal
# |chi|/2pi = 18.2 kHz; the cavity Ramsey of 23 October measured chi/2pi = -16.09 kHz.
DEVICE = dict(
    chi_Hz=18.2e3,              # design dispersive shift magnitude
    chi_measured_Hz=-16.09e3,   # direct cavity Ramsey (not used by the shipped designs)
    f_rabi_Hz=6e6,              # Omega_R/2pi; programmed with Omega_R_Hz = -6 MHz
    MHz_per_V_red=453.0,        # drive conversion the waveforms were built with
    MHz_per_V_blue=454.0,
    T1q=22.3e-6, T2star=9.2e-6, T2echo=28.0e-6, T1c=150e-6,
    grid_ns=4,                  # OPX waveform grid
)
SIM_OPTS = {"nsteps": 4000000, "atol": 1e-9, "rtol": 1e-8,
            "store_final_state": True, "store_states": False}
SCHED_KEYS = ('t_sb_ramp_ns', 't_pi2_ns', 't_rabi_ramp_ns', 't_pre_oc_ns', 't_down_ns')


def use_cpu_only():
    """Hide the GPU from TensorFlow (call before any TF work). On the Metal build the complex128
    sector propagation is ~10x slower with the GPU visible."""
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as e:          # devices already initialized
        print('use_cpu_only:', e)
    return tf.config.get_visible_devices()


def beat_ns(f_rabi_Hz=DEVICE['f_rabi_Hz']):
    """Beat-null spacing h = pi/Omega_R = 1/(2 f_R): 83.33 ns at 6 MHz."""
    return 1e9/(2*abs(f_rabi_Hz))


def on_grid(t_ns, grid=DEVICE['grid_ns']):
    return int(grid*round(t_ns/grid))


def beat_aligned(k, f_rabi_Hz=DEVICE['f_rabi_Hz'], grid=DEVICE['grid_ns']):
    """k beat intervals rounded to the 4 ns grid: 84, 168, 248, 332, 416, 500 ns for k = 1..6."""
    return on_grid(k*beat_ns(f_rabi_Hz), grid)


# ============================================================================ targets
_SQ = np.sqrt(np.pi/2)


def fock_vec(n, N):
    v = np.zeros(N, complex); v[n] = 1.0
    return v


def coherent_vec(alpha, N):
    n = np.arange(N); lf = np.array([0.5*lgamma(k+1) for k in n])
    if alpha == 0:
        return fock_vec(0, N)
    v = np.exp(-abs(alpha)**2/2 + n*np.log(abs(alpha)) - lf)*np.exp(1j*n*np.angle(alpha))
    return v/np.linalg.norm(v)


def cat_vec(alpha_cat, N, parity='even'):
    s = 1.0 if parity == 'even' else -1.0
    v = coherent_vec(alpha_cat, N) + s*coherent_vec(-alpha_cat, N)
    return v/np.linalg.norm(v)


def _gkp_lattice(N, kmax=12):
    lsf = np.array([0.5*lgamma(n+1) for n in range(N)]); n = np.arange(N)
    c = np.zeros(N, complex)
    for k in range(-kmax, kmax+1):
        for l in range(-kmax, kmax+1):
            z = 2*k*_SQ + 1j*l*_SQ; ph = np.exp(-1j*np.pi*k*l)
            if z == 0:
                coh = np.zeros(N, complex); coh[0] = 1.0
            else:
                coh = np.exp(-abs(z)**2/2 + n*np.log(abs(z)) - lsf)*np.exp(1j*n*np.angle(z))
            c += ph*coh
    return c


def gkp_vec(Delta, N):
    """Square-lattice GKP |0_L>, envelope exp(-Delta^2 n) (the mu = 0 branch used throughout)."""
    v = np.exp(-Delta**2*np.arange(N))*_gkp_lattice(N)
    return v/np.linalg.norm(v)


def make_target(kind, N=50, **kw):
    """kind 'fock' (n), 'cat' (alpha_cat, parity) or 'gkp' (Delta). Returns the cavity vector,
    its parity, and the extra keys the shipped target_data / hardware_params carry."""
    if kind == 'fock':
        n = int(kw.get('n', 2)); cav = fock_vec(n, N); name = f'Fock |{n}>'
        extra_t, extra_hw = {}, {}
    elif kind == 'cat':
        a = float(kw.get('alpha_cat', 1.5)); par = kw.get('parity', 'even')
        cav = cat_vec(a, N, par); name = f'{par} cat alpha={a:g}'
        extra_t, extra_hw = {'cat_alpha': a}, {'cat_alpha': a}
    elif kind == 'gkp':
        d = float(kw.get('Delta', 0.3401)); cav = gkp_vec(d, N); name = f'GKP |0_L> Delta={d:.4f}'
        extra_t = extra_hw = {'gkp_delta': d, 'gkp_mu': 0}
    else:
        raise ValueError(kind)
    p = np.abs(cav)**2
    pe = float(p[0::2].sum())
    assert pe > 1 - 1e-9 or pe < 1e-9, 'target must have definite parity to be reachable from |0,g>'
    parity = 'even' if pe > 0.5 else 'odd'
    return dict(kind=kind, name=name, cav=cav, parity=parity, extra_target=extra_t,
                extra_hw=extra_hw, spec=dict(kw))


def target_data(target):
    """target_data.npy content in the shipped format."""
    cav = np.asarray(target['cav'], complex)
    q = np.array([1.0, 0.0]) if target['parity'] == 'even' else np.array([0.0, 1.0])
    td = {'target_gkp_cavity': cav.reshape(-1, 1).astype(np.complex128),
          'target_full': np.kron(cav, q).reshape(len(cav), 2).astype(np.complex128),
          'parity': target['parity']}
    td.update(target['extra_target'])
    return td


# ============================================================================ characteristic function
def displacement(beta, N):
    """<m|D(beta)|n> from the Laguerre formula (no truncation artefact in the top rows)."""
    x = abs(beta)**2; lf = np.array([lgamma(k+1) for k in range(N)])
    M = np.zeros((N, N), complex)
    for m in range(N):
        for k in range(m+1):
            d = m - k; pref = np.exp(0.5*(lf[k] - lf[m]) - x/2)*eval_genlaguerre(k, d, x)
            M[m, k] = pref*beta**d
            if d:
                M[k, m] = pref*(-np.conj(beta))**d
    return M


def charfunc_grid(rho, xs):
    """Re C(beta) on the square grid xs x xs."""
    rho = np.asarray(rho)
    if rho.ndim == 1:
        rho = np.outer(rho, rho.conj())
    N = rho.shape[0]; LF = np.array([lgamma(k+1) for k in range(N)])
    B = (xs[None, :] + 1j*xs[:, None]).ravel(); x = np.abs(B)**2
    C = np.zeros(len(B), complex)
    for m in range(N):
        for k in range(m+1):
            d = m - k; pref = np.exp(0.5*(LF[k] - LF[m]) - x/2)*eval_genlaguerre(k, d, x)
            C += rho[k, m]*pref*B**d
            if d:
                C += rho[m, k]*pref*(-np.conj(B))**d
    return C.real.reshape(len(xs), len(xs))


def load_gkp_plan(folder=None):
    """32-point GKP estimator F_est = a0 + sum_j a_j Re C(beta_j) (gkp_charfunc_delta.ipynb, sec. 1).
    Reads gkp_measurement_plan.npy from a shipped GKP folder if one is given and present, else the
    CSV in data/ (weights to 5 digits, a0 from the plan)."""
    if folder is not None and os.path.exists(f'{folder}/gkp_measurement_plan.npy'):
        return np.load(f'{folder}/gkp_measurement_plan.npy', allow_pickle=True).item()
    rows = np.genfromtxt(ROOT/'data/gkp_points_Delta0340.csv', delimiter=',',
                         names=True)
    order = np.argsort(rows['j'])
    rows = rows[order]
    return dict(Delta=0.3401187062263489, pts=rows['beta_re'] + 1j*rows['beta_im'],
                a=rows['weight_a'].astype(float), a0=0.08555754406396407,
                C_target=rows['C_target'].astype(float), K=len(rows),
                note='F_est = a0 + sum_j a_j ReC(beta_j); post-selected cavity state; '
                     'beta in physical units (C(beta)=<D(beta)>, D=exp(beta a^dag - beta^* a))')


def charfunc_objective_points(target, plan=None):
    """Measurement points of the thesis characteristic-function objectives (Sec. 6.2)."""
    kind = target['kind']
    if kind == 'fock':          # 4 cuts x 25 radii up to |beta| = 3, plus an isotropy term
        th = np.arange(4)*np.pi/4; rr = np.linspace(0, 3.0, 25)
        pts = np.array([r*np.exp(1j*t) for t in th for r in rr]); W = np.ones(len(pts))
        return dict(pts=pts, W=W, n_radial=25, isotropy=True)
    if kind == 'cat':           # 4 cuts x 41 signed radii up to 4, weights on the coherence blobs
        a = float(target['spec'].get('alpha_cat', 1.5))
        th = np.arange(4)*np.pi/4; rr = np.linspace(-4, 4, 41)
        pts = np.array([r*np.exp(1j*t) for t in th for r in rr])
        w = 1 + 4*np.exp(-(np.abs(rr) - 2*a)**2/(2*0.5**2)); W = np.tile(w, 4)
        return dict(pts=pts, W=W/W.mean(), n_radial=41, isotropy=False)
    if kind == 'gkp':           # p_g * F_est from the 32-point plan
        plan = load_gkp_plan() if plan is None else plan
        if abs(plan['Delta'] - target['spec'].get('Delta', plan['Delta'])) > 1e-3:
            print('warning: the 32-point plan was designed for Delta=%.4f' % plan['Delta'])
        return dict(pts=np.asarray(plan['pts']), a=np.asarray(plan['a'], float), a0=float(plan['a0']))
    raise ValueError(kind)


# ============================================================================ 1. effective-H optimizer
def fourier_basis(T, n_steps, n_harmonics):
    """Phase samples at linspace(0, T, n_steps) with step T/n_steps, as in every shipped run."""
    t = np.linspace(0, T, n_steps); om = TWO_PI*np.arange(1, n_harmonics + 1)/T
    return (np.cos(np.outer(t, om)) - 1, np.sin(np.outer(t, om)),
            -om*np.sin(np.outer(t, om)), om*np.cos(np.outer(t, om)))


def sector_h0(alpha, chi, Nc):
    """H_eff(phi1 = phi2 = 0) in the conserved sector {|n, n mod 2>} (thesis App. A.1)."""
    lad = np.sqrt(np.arange(1, Nc))
    return -alpha*chi/2*(np.diag(lad, 1) + np.diag(lad, -1))


def sector_phases(Nc):
    n = np.arange(Nc)
    return np.ceil(n/2), np.floor(n/2)


def check_sector_reduction(alpha=17.0, chi=TWO_PI*DEVICE['chi_Hz'], Nc=20, dt=4e-9,
                           p1=0.37, p2=-0.51):
    """Max |V U0 V^dag - exp(-i dt H_full)| restricted to the sector (should be ~1e-15)."""
    a = np.diag(np.sqrt(np.arange(1, Nc)), 1); ac = np.kron(a, np.eye(2))
    sp = np.kron(np.eye(Nc), [[0, 0], [1, 0]]); sm = sp.T
    hfull = -alpha*chi/2*(np.exp(-1j*p1)*sm@ac + np.exp(1j*p2)*sm@ac.T
                          + np.exp(-1j*p2)*sp@ac + np.exp(1j*p1)*sp@ac.T)
    sec = 2*np.arange(Nc) + np.arange(Nc) % 2
    n1, n2 = sector_phases(Nc); v = np.exp(1j*(n1*p1 + n2*p2))
    red = v[:, None]*expm(-1j*dt*sector_h0(alpha, chi, Nc))*v.conj()[None, :]
    return float(np.abs(red - expm(-1j*dt*hfull)[np.ix_(sec, sec)]).max())


def effH_evolve(coeffs, alpha, T, n_steps, Nc=50, chi=TWO_PI*DEVICE['chi_Hz']):
    """Final sector state for one coefficient set {C1,S1,C2,S2: (n_harmonics,)}."""
    nh = len(coeffs['C1'])
    Bc, Bs, _, _ = fourier_basis(T, n_steps, nh)
    ph1 = Bc@np.asarray(coeffs['C1'], float) + Bs@np.asarray(coeffs['S1'], float)
    ph2 = Bc@np.asarray(coeffs['C2'], float) + Bs@np.asarray(coeffs['S2'], float)
    U0 = expm(-1j*T/n_steps*sector_h0(alpha, chi, Nc)); n1, n2 = sector_phases(Nc)
    psi = fock_vec(0, Nc)
    for i in range(n_steps):
        v = np.exp(1j*(n1*ph1[i] + n2*ph2[i]))
        psi = v*(U0@(v.conj()*psi))
    return psi


def effH_metrics(psi, target):
    """Joint fidelity F, herald probability p and post-selected cavity fidelity F/p."""
    cav = np.zeros(len(psi), complex); m = min(len(psi), len(target['cav']))
    cav[:m] = target['cav'][:m]; cav /= np.linalg.norm(cav)
    keep = (np.arange(len(psi)) % 2) == (0 if target['parity'] == 'even' else 1)
    p = float(np.sum(np.abs(psi[keep])**2)); F = float(abs(np.vdot(cav, psi))**2)
    return dict(F_joint=F, p_herald=p, F_cav=F/p)


def phase_derivative_max_MHz(coeffs, T, n_steps=2001):
    nh = len(coeffs['C1']); _, _, dBc, dBs = fourier_basis(T, n_steps, nh)
    d1 = dBc@np.asarray(coeffs['C1'], float) + dBs@np.asarray(coeffs['S1'], float)
    d2 = dBc@np.asarray(coeffs['C2'], float) + dBs@np.asarray(coeffs['S2'], float)
    return float(max(np.abs(d1).max(), np.abs(d2).max())/(TWO_PI*1e6))


class EffHOptimizer:
    """Batch of independent random starts, optimized with Adam on the effective Hamiltonian.

    objective = 'fidelity': -log F_joint (the full-fidelity loss, Eq. 6.2);
                'charfunc': the measurable objectives of Eqs. 6.3-6.5 (Fock: 4 cuts + isotropy;
                            cat: blob-weighted 4 cuts; GKP: -log p_g F_est at 32 points).
    Both add R = 1e-4 <k (C^2 + S^2)> + 15/phidot_max^2 <ReLU(|phidot| - phidot_max)^2>.
    Propagation is exact piecewise-constant in the sector {|n, n mod 2>}: one precomputed matrix
    exponential and a diagonal phase change per step (thesis App. A.1)."""

    def __init__(self, target, alpha, objective='fidelity', T=2e-6, n_steps=500, n_harmonics=20,
                 batch=3, lr=1e-2, max_deriv_MHz=5.0, deriv_weight=15.0, reg_weight=1e-4,
                 chi=TWO_PI*DEVICE['chi_Hz'], seed=42, init_scale=0.3, warm_start=None, plan=None):
        import tensorflow as tf
        self.tf = tf
        self.target, self.alpha, self.objective = target, float(alpha), objective
        self.T, self.n_steps, self.nh, self.B = float(T), int(n_steps), int(n_harmonics), int(batch)
        self.chi, self.seed = float(chi), seed
        self.Nc = Nc = len(target['cav'])
        self.max_deriv = TWO_PI*max_deriv_MHz*1e6
        self.deriv_weight, self.reg_weight = float(deriv_weight), float(reg_weight)
        rng = np.random.default_rng(seed)       # same draw order as the original optimizer
        init = {k: rng.normal(0.0, init_scale, (self.B, self.nh)) for k in ('C1', 'S1', 'C2', 'S2')}
        if warm_start is not None:
            for k, v in warm_start.items():
                init[k] = np.broadcast_to(np.asarray(v, float), (self.B, self.nh)).copy()
        self.device = tf.device('/CPU:0')     # complex128 has no Metal/GPU kernels; the sector is small
        with self.device:
            self.vars = {k: tf.Variable(v, dtype=tf.float64, name=k) for k, v in init.items()}
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

        c64 = lambda x: tf.constant(np.asarray(x), tf.float64)
        self.basis = [c64(x) for x in fourier_basis(self.T, self.n_steps, self.nh)]
        self.U0 = tf.constant(expm(-1j*self.T/self.n_steps*sector_h0(self.alpha, self.chi, Nc)),
                              tf.complex128)
        n1, n2 = sector_phases(Nc); self.n1, self.n2 = c64(n1), c64(n2)
        par = 0 if target['parity'] == 'even' else 1
        self.herald = tf.constant(((np.arange(Nc) % 2) == par).astype(complex), tf.complex128)
        self.other = tf.constant(((np.arange(Nc) % 2) != par).astype(complex), tf.complex128)
        self.cav = tf.constant(target['cav'], tf.complex128)
        psi0 = np.zeros((self.B, Nc), complex); psi0[:, 0] = 1
        self.psi0 = tf.constant(psi0, tf.complex128)
        self.cf = None
        if objective == 'charfunc':
            cf = charfunc_objective_points(target, plan)
            D = np.stack([displacement(b, Nc) for b in cf['pts']])
            self.cf = cf
            self.D = tf.constant(D, tf.complex128)
            if target['kind'] == 'gkp':
                M = cf['a0']*np.eye(Nc) + np.einsum('k,kij->ij', cf['a'],
                                                     (D + D.conj().transpose(0, 2, 1))/2)
                self.M = tf.constant(M, tf.complex128)
            else:
                cav = target['cav']
                self.Ct = tf.constant(np.einsum('j,mjk,k->m', cav.conj(), D, cav), tf.complex128)
                self.W = tf.constant(cf['W'], tf.float64)
        elif objective != 'fidelity':
            raise ValueError(objective)
        self._train = tf.function(self._train_step)

    # -------------------------------------------------------------------------------------
    def _phases(self):
        tf = self.tf; Bc, Bs, dBc, dBs = self.basis; v = self.vars
        ph1 = v['C1'] @ tf.transpose(Bc) + v['S1'] @ tf.transpose(Bs)       # (B, steps)
        ph2 = v['C2'] @ tf.transpose(Bc) + v['S2'] @ tf.transpose(Bs)
        pd1 = v['C1'] @ tf.transpose(dBc) + v['S1'] @ tf.transpose(dBs)
        pd2 = v['C2'] @ tf.transpose(dBc) + v['S2'] @ tf.transpose(dBs)
        return ph1, ph2, pd1, pd2

    def _terms(self):
        tf = self.tf
        ph1, ph2, pd1, pd2 = self._phases()
        ang = tf.transpose(ph1[:, :, None]*self.n1 + ph2[:, :, None]*self.n2, [1, 0, 2])
        V = tf.exp(tf.complex(tf.zeros_like(ang), ang))                       # (steps, B, Nc)
        U0 = self.U0
        psi = tf.scan(lambda s, v: v*tf.einsum('ij,bj->bi', U0, tf.math.conj(v)*s), V,
                      initializer=self.psi0, parallel_iterations=1)[-1]
        F = tf.abs(tf.einsum('i,bi->b', tf.math.conj(self.cav), psi))**2
        hb = psi*self.herald
        p = tf.reduce_sum(tf.abs(hb)**2, axis=1)
        if self.objective == 'fidelity':
            data = -tf.math.log(tf.maximum(F, 1e-12))
        elif self.target['kind'] == 'gkp':
            score = tf.math.real(tf.einsum('bi,ij,bj->b', tf.math.conj(hb), self.M, hb))
            data = -tf.math.log(tf.maximum(score, 1e-3))
        else:
            ob = psi*self.other                                    # traced cavity state
            C = (tf.einsum('bj,mjk,bk->bm', tf.math.conj(hb), self.D, hb)
                 + tf.einsum('bj,mjk,bk->bm', tf.math.conj(ob), self.D, ob))
            ce = tf.reduce_sum(self.W[None, :]*tf.abs(C - self.Ct[None, :])**2, axis=1)/tf.reduce_sum(self.W)
            data = 100.0*ce + 50.0*(1.0 - p)**2
            if self.cf['isotropy']:
                Cr = tf.reshape(C, [self.B, 4, self.cf['n_radial']])
                data += 50.0*tf.reduce_mean(tf.abs(Cr[:, 0] - Cr[:, 2])**2
                                            + tf.abs(Cr[:, 1] - Cr[:, 3])**2, axis=1)
        k = tf.range(1, self.nh + 1, dtype=tf.float64)[None, :]
        reg = self.reg_weight*tf.add_n([tf.reduce_mean(k*v**2) for v in self.vars.values()])
        md = self.max_deriv
        reg += self.deriv_weight*tf.reduce_mean(tf.nn.relu(tf.abs(pd1) - md)**2
                                                + tf.nn.relu(tf.abs(pd2) - md)**2)/md**2
        return data, F, p, reg

    def _train_step(self):
        tv = list(self.vars.values())
        with self.tf.GradientTape() as tape:
            data, F, p, reg = self._terms()
            L = self.tf.reduce_mean(data) + reg
        self.opt.apply_gradients(zip(tape.gradient(L, tv), tv))
        return L, data, F, p

    # -------------------------------------------------------------------------------------
    def optimize(self, iters=300, every=50, stop_at=None, verbose=True):
        """Adam steps; the best member (lowest data loss, pre-update parameters) is kept."""
        hist = getattr(self, 'hist', {k: [] for k in ('loss', 'obj', 'F_joint', 'p_herald', 'F_joint_max')})
        best = getattr(self, 'best', dict(obj=np.inf))
        t0 = time.time()
        for it in range(iters):
            cur = {k: v.numpy().copy() for k, v in self.vars.items()}
            with self.device:
                L, data, F, p = (x.numpy() for x in self._train())
            k = int(np.argmin(data))
            for key, val in (('loss', L), ('obj', data[k]), ('F_joint', F[k]), ('p_herald', p[k]),
                             ('F_joint_max', F.max())):
                hist[key].append(float(val))
            if data[k] < best['obj']:
                best = dict(obj=float(data[k]), F_joint=float(F[k]), p_herald=float(p[k]), idx=k,
                            iteration=len(hist['loss']) - 1, coeffs={kk: vv[k] for kk, vv in cur.items()})
            if verbose and (it % every == 0 or it == iters - 1):
                print('it %4d  loss %8.4f | best member: F_joint %.4f  p %.4f | F_joint max %.4f  (%.0f s)'
                      % (len(hist['loss']) - 1, L, F[k], p[k], F.max(), time.time() - t0), flush=True)
            if stop_at is not None and F.max() >= stop_at:
                break
        self.hist, self.best = hist, best
        return best, hist

    def solution(self):
        """The best coefficients with their (numpy) effective-model metrics."""
        c = self.best['coeffs']
        m = effH_metrics(effH_evolve(c, self.alpha, self.T, self.n_steps, self.Nc, self.chi), self.target)
        return dict(coeffs=c, alpha=self.alpha, T=self.T, n_steps=self.n_steps, n_harmonics=self.nh,
                    Nc=self.Nc, chi=self.chi, target=self.target, history=dict(self.hist),
                    objective=self.objective, best_fidelity=m['F_joint'], phidot_max_MHz=
                    phase_derivative_max_MHz(c, self.T), **m)


def load_effH_source(folder, batch_index=0, target=None):
    """An effective-model solution in the 4-file format (optimizer/results/target/controls_data)."""
    o = np.load(f'{folder}/optimizer_data.npy', allow_pickle=True).item()
    r = np.load(f'{folder}/results_data.npy', allow_pickle=True).item()
    t = np.load(f'{folder}/target_data.npy', allow_pickle=True).item()
    c = {k: np.asarray(r['best_coeffs'][k][batch_index], float) for k in ('C1', 'S1', 'C2', 'S2')}
    if target is None:
        cav = np.asarray(t['target_gkp_cavity']).ravel()
        p = np.abs(cav)**2
        target = dict(kind='stored', name=folder, cav=cav, spec={},
                      parity=t.get('parity', 'even' if p[0::2].sum() > 0.5 else 'odd'),
                      extra_target={k: v for k, v in t.items()
                                    if k not in ('target_gkp_cavity', 'target_full', 'parity')},
                      extra_hw={})
    sol = dict(coeffs=c, alpha=float(o['alpha']), T=float(o['total_time']), n_steps=int(o['n_steps']),
               n_harmonics=int(o['n_harmonics']), Nc=int(o['Nc']), chi=float(o['chi']), target=target,
               history=r.get('history', {}), objective=r.get('objective', ''),
               best_fidelity=float(r['best_fidelity']), stored_target_data=t, source=folder)
    sol.update(effH_metrics(effH_evolve(c, sol['alpha'], sol['T'], sol['n_steps'], sol['Nc'], sol['chi']),
                            target))
    return sol


def save_effH_source(folder, sol):
    """Write an effective-model solution in the 4-file format that load_effH_source reads."""
    os.makedirs(folder, exist_ok=True)
    Bc, Bs, dBc, dBs = fourier_basis(sol['T'], sol['n_steps'], sol['n_harmonics'])
    c = sol['coeffs']
    np.save(f'{folder}/optimizer_data.npy', {
        'Nc': int(sol['Nc']), 'Nq': 2, 'n_steps': int(sol['n_steps']), 'total_time': float(sol['T']),
        'n_harmonics': int(sol['n_harmonics']), 'alpha': float(sol['alpha']), 'chi': float(sol['chi']),
        'g_MHz': sol['alpha']*sol['chi']/TWO_PI/2/1e6,
        'Bcos': Bc.astype(np.float32), 'Bsin': Bs.astype(np.float32)}, allow_pickle=True)
    np.save(f'{folder}/results_data.npy', {
        'best_coeffs': {k: np.asarray(c[k], float)[None, :] for k in ('C1', 'S1', 'C2', 'S2')},
        'best_fidelity': float(sol['F_joint']), 'history': sol.get('history', {}),
        'p_g': float(sol['p_herald']), 'F_cav': float(sol['F_cav']), 'objective': sol.get('objective', '')},
        allow_pickle=True)
    np.save(f'{folder}/target_data.npy', target_data(sol['target']), allow_pickle=True)
    np.save(f'{folder}/controls_data.npy', {
        'time_ns': np.linspace(0, sol['T'], sol['n_steps'])*1e9,
        'phi1': Bc@c['C1'] + Bs@c['S1'], 'phi2': Bc@c['C2'] + Bs@c['S2'],
        'phi1_dot': dBc@c['C1'] + dBs@c['S1'], 'phi2_dot': dBc@c['C2'] + dBs@c['S2'],
        'alpha': float(sol['alpha']), 'chi': float(sol['chi'])}, allow_pickle=True)


# ============================================================================ 2. export to hardware
def _phase_arrays(coeffs, T, n_pts):
    """Sim pairing used by every shipped file: phi1 <- stored (C2, S2), phi2 <- stored (C1, S1).
    The hardware generators do the same swap through Omega_R_Hz < 0."""
    nh = len(coeffs['C1'])
    tt = np.linspace(0, T, n_pts); om = TWO_PI*np.arange(1, nh + 1)/T
    Bc = np.cos(np.outer(tt, om)) - 1.0; Bs = np.sin(np.outer(tt, om))
    dBc = -om*np.sin(np.outer(tt, om)); dBs = om*np.cos(np.outer(tt, om))
    C1, S1 = np.asarray(coeffs['C2'], float), np.asarray(coeffs['S2'], float)
    C2, S2 = np.asarray(coeffs['C1'], float), np.asarray(coeffs['S1'], float)
    return tt, Bc, Bs, Bc@C1 + Bs@S1, Bc@C2 + Bs@S2, dBc@C1 + dBs@S1, dBc@C2 + dBs@S2


def rescale_alpha(alpha_opt, T_opt, T_oc_ns):
    """alpha_hw = alpha_opt T_opt / T_oc; the Fourier coefficients are unchanged."""
    return alpha_opt*T_opt/(T_oc_ns*1e-9)


def check_schedule(sched, T_oc_ns, f_rabi_Hz=DEVICE['f_rabi_Hz'], verbose=True):
    """4 ns grid, T_oc a multiple of 6h, and the offset of every boundary from a beat null."""
    h = beat_ns(f_rabi_Hz); t_sb, t_pi2, t_rr, t_pre, t_dn = sched
    seg = dict(t_sb=t_sb, t_pi2=t_pi2, t_rr=t_rr, t_pre=t_pre, T_oc=T_oc_ns, t_dn=t_dn)
    for k, v in seg.items():
        assert int(v) == v and v % DEVICE['grid_ns'] == 0, f'{k}={v} ns is not on the 4 ns grid'
    assert T_oc_ns % int(round(6*h)) == 0, 'T_oc must be a multiple of 6h = %d ns' % round(6*h)
    bounds = np.cumsum([0, t_sb, t_pi2, t_rr, t_pre, T_oc_ns, t_dn])
    off = [(b/h - round(b/h))*h for b in bounds]
    if verbose:
        print('schedule  sb %d | pi2 %d | rabi %d | pre %d | OC %d | down %d | pi2 %d   (total %.3f us)'
              % (t_sb, t_pi2, t_rr, t_pre, T_oc_ns, t_dn, t_pi2, (bounds[-1] + t_pi2)/1e3))
        print('  boundary times (ns): ' + ' '.join('%d' % b for b in bounds))
        print('  offset from nearest beat null k*h (ns): ' + ' '.join('%+.1f' % o for o in off))
    return np.array(off)


def export_hardware(sol, out, T_oc_ns, sched, Nc_sim, f_rabi_Hz=DEVICE['f_rabi_Hz'], note='',
                    plan=None, verbose=True):
    """Write a measurement-ready folder (convert_to_full_H of the shipped conversion):
    final_coeffs, final_optimizer, target_data, final_history, controls_data, a provisional
    hardware_params (completed by finalize_hardware) and, for GKP, gkp_measurement_plan."""
    nh = int(sol['n_harmonics']); CHI = sol['chi']/TWO_PI; f_MHz = abs(f_rabi_Hz)/1e6
    T = T_oc_ns*1e-9; AL = rescale_alpha(sol['alpha'], sol['T'], T_oc_ns)
    os.makedirs(out, exist_ok=True)
    tt, Bc, Bs, p1, p2, d1, d2 = _phase_arrays(sol['coeffs'], T, T_oc_ns + 1)
    p2 = p2 + np.pi
    np.save(f'{out}/final_coeffs.npy', {k: np.asarray(sol['coeffs'][k], np.float32)[None, :]
                                        for k in ('C1', 'S1', 'C2', 'S2')}, allow_pickle=True)
    np.save(f'{out}/final_optimizer.npy', {
        'Nc': int(sol['Nc']), 'n_steps': int(T_oc_ns), 'total_time': T, 'n_harmonics': nh,
        'alpha': float(AL), 'chi': float(TWO_PI*CHI), 'Omega_R': float(TWO_PI*f_MHz*1e6),
        'best_fidelity': float(sol['best_fidelity']),
        'Bcos': Bc.astype(np.float32), 'Bsin': Bs.astype(np.float32)}, allow_pickle=True)
    td = sol.get('stored_target_data') or target_data(sol['target'])
    np.save(f'{out}/target_data.npy', td, allow_pickle=True)
    np.save(f'{out}/final_history.npy', sol.get('history', {}), allow_pickle=True)
    np.save(f'{out}/controls_data.npy', {
        'time_ns': tt*1e9, 'phi1': p1, 'phi2': p2 - np.pi, 'phi1_dot': d1, 'phi2_dot': d2,
        'alpha': float(AL), 'chi': float(TWO_PI*CHI), 'Omega_R': float(TWO_PI*f_MHz*1e6)},
        allow_pickle=True)
    parity = sol['target']['parity']
    t_sb, t_pi2, t_rr, t_pre, t_dn = (int(x) for x in sched)
    hw = {'f_rabi_Hz': f_MHz*1e6, 'Omega_R_Hz': -f_MHz*1e6, 'target_parity': parity,
          'pi2_close_sign': -1.0 if parity == 'odd' else 1.0,
          't_sb_ramp_ns': t_sb, 't_pi2_ns': t_pi2, 't_rabi_ramp_ns': t_rr, 't_pre_oc_ns': t_pre,
          't_down_ns': t_dn, 'rabi_ramp_ns': t_rr + t_pi2, 'phi_plus': np.pi, 'phi_minus': 0.0,
          'ff_mode': 'timedep', 'chi_Hz': CHI, 'alpha': float(AL), 'Nc_sim': int(Nc_sim),
          'T_oc_ns': int(T_oc_ns), 'h_ns': 500.0/f_MHz}
    hw.update(sol['target'].get('extra_hw', {}))
    np.save(f'{out}/hardware_params.npy', hw, allow_pickle=True)
    if sol['target']['kind'] == 'gkp' or 'gkp_delta' in td:
        np.save(f'{out}/gkp_measurement_plan.npy', plan if plan is not None else load_gkp_plan(),
                allow_pickle=True)
    W = -abs(TWO_PI*f_rabi_Hz)
    Vp = AL*(W + d1)/(TWO_PI*1e6)/DEVICE['MHz_per_V_red']
    Vm = AL*(W - d2)/(TWO_PI*1e6)/DEVICE['MHz_per_V_blue']
    wf = Vp*np.exp(1j*p1)*np.exp(1j*W*tt) + Vm*np.exp(1j*p2)*np.exp(-1j*W*tt)
    info = dict(alpha_hw=float(AL), peak_V=float(np.abs(wf).max()), mean_V=float(np.abs(wf).mean()),
                peak_n=(2*AL)**2, phidot_max_MHz=float(max(np.abs(d1).max(), np.abs(d2).max())/(TWO_PI*1e6)))
    if verbose:
        print('wrote %s: alpha_hw = %.4f (alpha_opt %.3g x %.2f us / %.2f us), peak |V| = %.3f V, '
              'peak <n> = %.0f, max |phidot| = %.2f MHz, parity %s'
              % (out, AL, sol['alpha'], sol['T']*1e6, T*1e6, info['peak_V'], info['peak_n'],
                 info['phidot_max_MHz'], parity))
    return info


def finalize_hardware(out, sched, F, pg, Nc, note='', frame_deg=None, verbose=True):
    """Schedule and Stark feed-forward numbers into hardware_params (finalize of the shipped
    conversion), with the invariants asserted."""
    o = np.load(f'{out}/final_optimizer.npy', allow_pickle=True).item()
    c = np.load(f'{out}/final_coeffs.npy', allow_pickle=True).item()
    hw = dict(np.load(f'{out}/hardware_params.npy', allow_pickle=True).item())
    t_sb, t_pi2, t_rr, t_pre, t_dn = (int(x) for x in sched)
    al, chi, nh, T = o['alpha'], o['chi'], int(o['n_harmonics']), o['total_time']
    n_oc = int(round(T*1e9)); Om = -abs(TWO_PI*hw['f_rabi_Hz'])
    total = t_sb + t_pi2 + t_rr + t_pre + n_oc + t_dn
    env = np.zeros(total); env[:t_sb] = np.sin(np.pi/2*np.linspace(0, 1, t_sb))**2
    env[t_sb:total - t_dn] = 1.0
    env[total - t_dn:] = np.sin(np.pi/2*(1 - np.linspace(0, 1, t_dn)))**2
    ph1 = np.zeros(total); ph2 = np.full(total, np.pi)
    _, _, _, q1, q2, _, _ = _phase_arrays({k: c[k][0] for k in c}, T, n_oc)
    i0 = t_sb + t_pi2 + t_rr + t_pre
    ph1[i0:i0 + n_oc] = q1; ph2[i0:i0 + n_oc] = q2 + np.pi
    ph1[i0 + n_oc:] = ph1[i0 + n_oc - 1]; ph2[i0 + n_oc:] = ph2[i0 + n_oc - 1]
    tg = np.arange(total)*1e-9
    beta = env*al*(np.exp(1j*(Om*tg + ph1)) + np.exp(-1j*(Om*tg - ph2)))
    dl = -chi/2*np.abs(beta)**2/TWO_PI; r0 = t_sb + t_pi2
    hw.update({'t_sb_ramp_ns': t_sb, 't_pi2_ns': t_pi2, 't_rabi_ramp_ns': t_rr,
               't_pre_oc_ns': t_pre, 't_down_ns': t_dn, 'rabi_ramp_ns': t_rr + t_pi2,
               'sideband_wf_len_ns': int(total), 'rabi_oc_wf_len_ns': int(t_rr + t_pre + n_oc + t_dn),
               'oc_start_ns': int(i0), 'T_oc_ns': int(n_oc), 'alpha': float(al), 'Nc_sim': int(Nc),
               'F_sim': round(float(F), 4), 'p_g_sim': round(float(pg), 4),
               'delta_mean_Hz': float(dl[r0:].mean()), 'delta_mean_OC_Hz': float(dl[i0:i0 + n_oc].mean()),
               'delta_pi2_Hz': float(dl[t_sb:t_sb + t_pi2].mean()), 'delta_max_Hz': float(np.abs(dl).max()),
               'h_ns': 500.0/(hw['f_rabi_Hz']/1e6), 'note': note})
    if frame_deg is not None:
        hw['frame_deg_sim'] = round(float(frame_deg), 2)
    assert hw['rabi_ramp_ns'] == hw['t_rabi_ramp_ns'] + hw['t_pi2_ns']
    assert hw['sideband_wf_len_ns'] == (hw['t_sb_ramp_ns'] + hw['t_pi2_ns'] + hw['t_rabi_ramp_ns']
                                        + hw['t_pre_oc_ns'] + hw['T_oc_ns'] + hw['t_down_ns'])
    assert hw['rabi_oc_wf_len_ns'] == (hw['t_rabi_ramp_ns'] + hw['t_pre_oc_ns'] + hw['T_oc_ns']
                                       + hw['t_down_ns'])
    for k in ('t_sb_ramp_ns', 't_rabi_ramp_ns', 't_pre_oc_ns', 't_down_ns', 'T_oc_ns',
              'rabi_ramp_ns', 'sideband_wf_len_ns', 'rabi_oc_wf_len_ns', 'oc_start_ns'):
        assert hw[k] % DEVICE['grid_ns'] == 0, '%s=%d is not on the 4 ns AWG grid' % (k, hw[k])
    assert hw['T_oc_ns'] % int(round(6*hw['h_ns'])) == 0, 'T_oc must be a multiple of 6h'
    np.save(f'{out}/hardware_params.npy', hw, allow_pickle=True)
    if verbose:
        print('wrote %s/hardware_params.npy: F_sim %.4f  p_g_sim %.4f  delta_mean_OC %.3f MHz  '
              'delta_pi2 %.3f MHz' % (out, hw['F_sim'], hw['p_g_sim'], hw['delta_mean_OC_Hz']/1e6,
                                      hw['delta_pi2_Hz']/1e6))
    return hw


def load_folder(folder):
    return {f[:-4]: np.load(f'{folder}/{f}', allow_pickle=True).item()
            for f in sorted(os.listdir(folder)) if f.endswith('.npy')}


def compare_folders(new, ref, rtol=1e-5, skip=('note', 'F_sim', 'p_g_sim', 'frame_deg_sim', 'Nc_sim')):
    """File-by-file, key-by-key comparison of two solution folders; returns the worst relative
    difference per file. Keys in `skip` are printed side by side, not compared."""
    A, B = load_folder(new), load_folder(ref)
    worst = {}
    for f in sorted(set(A) | set(B)):
        if f not in A or f not in B:
            print('%-22s only in %s' % (f, 'new' if f in A else 'reference')); continue
        a, b = A[f], B[f]; ka, kb = set(a), set(b)
        if ka != kb:
            print('%-22s keys differ: new-only %s, reference-only %s' % (f, sorted(ka - kb), sorted(kb - ka)))
        w = 0.0
        for k in sorted(ka & kb):
            x, y = a[k], b[k]
            if k in skip:
                continue
            if isinstance(x, str) or isinstance(y, str):
                if x != y:
                    print('  %s[%s]: %r vs %r' % (f, k, x, y))
                continue
            if isinstance(x, dict) or isinstance(x, (list, tuple)):
                continue
            x, y = np.asarray(x), np.asarray(y)
            if x.dtype == bool or y.dtype == bool:
                x, y = x.astype(float), y.astype(float)
            if x.shape != y.shape:
                print('  %s[%s]: shape %s vs %s' % (f, k, x.shape, y.shape)); w = np.inf; continue
            d = float(np.max(np.abs(x - y))/max(np.max(np.abs(y)), 1e-30)) if x.size else 0.0
            w = max(w, d)
            if d > rtol:
                print('  %s[%s]: max rel diff %.2e' % (f, k, d))
        worst[f] = w
    print('max relative difference per file:', {k: '%.1e' % v for k, v in worst.items()})
    shown = [k for k in skip if k in A.get('hardware_params', {}) or k in B.get('hardware_params', {})]
    for k in shown:
        x, y = A['hardware_params'].get(k), B['hardware_params'].get(k)
        if isinstance(x, str) or isinstance(y, str):
            print('  hardware_params[%s]: %s' % (k, 'identical' if x == y else 'differs (provenance text)'))
        else:
            print('  hardware_params[%s]: new %r | reference %r' % (k, x, y))
    return worst


def hardware_waveforms(folder, rabi_amp=1.0, MHz_per_V_red=DEVICE['MHz_per_V_red'],
                       MHz_per_V_blue=DEVICE['MHz_per_V_blue']):
    """Build the two waveforms with the hardware-side generators (make_two_sideband_amp_oc.py,
    make_rabi_oc_amp.py, unchanged) from a solution folder, as oc_cell_from_params.py does.
    rabi_amp is the calibrated Rabi amplitude on the OPX (1.0 here: envelope only)."""
    ns = {'np': np, 'a_memory_m2_rs': MHz_per_V_red, 'a_memory_m2_bs': MHz_per_V_blue}
    for f in ('make_two_sideband_amp_oc.py', 'make_rabi_oc_amp.py'):
        exec(compile(open(ROOT/f).read(), str(ROOT/f), 'exec'), ns)
    hw = np.load(f'{folder}/hardware_params.npy', allow_pickle=True).item()
    sb = ns['make_two_sideband_amp_oc'](
        folder, Omega_R_Hz=hw['Omega_R_Hz'], ramp_ns=hw['t_sb_ramp_ns'], ramp_down_ns=hw['t_down_ns'],
        rabi_ramp_ns=hw['rabi_ramp_ns'], phi_plus=hw['phi_plus'], phi_minus=hw['phi_minus'])
    rb, info = ns['make_rabi_oc_amp'](
        folder, Omega_R_Hz=hw['Omega_R_Hz'], rabi_amp=rabi_amp, ramp_ns=hw['t_sb_ramp_ns'],
        rabi_ramp_ns=hw['rabi_ramp_ns'], ramp_down_ns=hw['t_down_ns'],
        rabi_ramp_up_ns=hw['t_rabi_ramp_ns'], t_pi2_ns=hw['t_pi2_ns'], T_PRE=hw['t_pre_oc_ns'],
        phi_plus=hw['phi_plus'], phi_minus=hw['phi_minus'], chi_Hz=hw['chi_Hz'], return_info=True)
    return sb, rb, info, hw


def keys_read_by(path, var='hw'):
    """hardware_params keys that a hardware-side cell reads as var['...']."""
    import re
    return sorted(set(re.findall(r"%s\['([A-Za-z0-9_]+)'\]" % var, open(path).read())))


# ============================================================================ 3. displaced-frame simulation
def build_sequence(folder, sched=None, OS=4):
    """Full sequence sb ramp | pi/2 | Rabi ramp | pre | OC | ramp down | pi/2 on a 1/OS ns grid.
    beta is the exact classical response, beta = -i int eps_played dt, from vacuum; the qubit
    sees it in the frame displaced by beta (stark_sim.py, approx_vs_alpha.py)."""
    o = np.load(f'{folder}/final_optimizer.npy', allow_pickle=True).item()
    c = np.load(f'{folder}/final_coeffs.npy', allow_pickle=True).item()
    hw = np.load(f'{folder}/hardware_params.npy', allow_pickle=True).item()
    al, chi, T = o['alpha'], o['chi'], o['total_time']
    Om = -abs(TWO_PI*hw['f_rabi_Hz'])
    sched = tuple(int(hw[k]) for k in SCHED_KEYS) if sched is None else tuple(int(x) for x in sched)
    ts, tp2, trr, tp, td = sched
    n_oc = int(round(T*1e9)); NS = lambda x: int(round(x*OS))
    ts_, tp2_, trr_, tp_, td_, noc_ = NS(ts), NS(tp2), NS(trr), NS(tp), NS(td), NS(n_oc)
    tdn = ts_ + tp2_ + trr_ + tp_ + noc_ + td_; NTOT = tdn + tp2_; NT = NTOT + 1
    i0, r0 = ts_ + tp2_ + trr_ + tp_, ts_ + tp2_
    up = lambda m: np.sin(np.pi/2*np.linspace(0, 1, m + 1))**2
    esb = np.zeros(NT); esb[:ts_ + 1] = up(ts_); esb[ts_:tdn - td_ + 1] = 1.0
    esb[tdn - td_:tdn + 1] = up(td_)[::-1]
    er = np.zeros(NT); er[r0:r0 + trr_ + 1] = up(trr_); er[r0 + trr_:tdn - td_ + 1] = 1.0
    er[tdn - td_:tdn + 1] = up(td_)[::-1]

    def rect(i_s, i_e):
        w = np.zeros(NT); w[i_s:i_e + 1] = 1.0; w[i_s] = w[i_e] = 0.5
        return w
    ph1 = np.zeros(NT); ph2 = np.full(NT, np.pi); pd1 = np.zeros(NT); pd2 = np.zeros(NT)
    _, _, _, q1, q2, e1_, e2_ = _phase_arrays({k: c[k][0] for k in c}, T, noc_ + 1)
    sl = slice(i0, i0 + noc_ + 1)
    ph1[sl] = q1; ph2[sl] = q2 + np.pi; pd1[sl] = e1_; pd2[sl] = e2_
    ph1[i0 + noc_:] = ph1[i0 + noc_]; ph2[i0 + noc_:] = ph2[i0 + noc_]
    tg = np.arange(NT)*(1e-9/OS)
    ex1 = np.exp(1j*(Om*tg + ph1)); ex2 = np.exp(-1j*(Om*tg - ph2))
    f = esb*(-al*(Om + pd1)*ex1 + al*(Om - pd2)*ex2)
    beta = -1j*np.concatenate([[0], np.cumsum(0.5*(f[1:] + f[:-1]))*(tg[1] - tg[0])])
    bdes = esb*al*(ex1 + ex2)
    sig = -chi/2*np.abs(bdes)**2; sig[tdn + 1:] = 0.0      # sigma_z coefficient that cancels the shift
    return dict(folder=folder, hw=hw, sched=sched, al=al, chi=chi, Om=Om, OS=OS, tg=tg, NT=NT,
                idx=dict(pi2_open=(ts_, ts_ + tp2_), rabi=r0, oc=(i0, i0 + noc_), down=tdn - td_, end=tdn,
                         pi2_close=(tdn, NTOT)),
                esb=esb, er=er, p2o=rect(ts_, ts_ + tp2_), p2c=rect(tdn, NTOT), beta=beta, bdes=bdes,
                sig=sig, Om2=(np.pi/2)/(tp2*1e-9), pi2c_sign=float(hw.get('pi2_close_sign', 1.0)))


def stark_ff(S, mode='td', scale=1.0):
    """sigma_z feed-forward. 'td': follows -chi/2 |beta(t)|^2 (design intent; make_rabi_oc_amp puts
    it in the rabi_oc phase). 'const': the segment averages, over the opening pi/2 and over the rest
    (delta_pi2_Hz / delta_mean_Hz). scale = 1 is the full correction; 0.5 is what playing the file
    number as a drive-frequency detuning applies (the initial hardware convention, thesis Ch. 7, 10).
    Off during the closing pi/2."""
    if mode == 'td':
        return scale*S['sig']
    if mode != 'const':
        raise ValueError(mode)
    a, b = S['idx']['pi2_open']; e = S['idx']['end']
    ff = np.zeros(S['NT'])
    ff[a:b] = scale*S['sig'][a:b].mean(); ff[b:e + 1] = scale*S['sig'][b:e + 1].mean()
    return ff


def _target_at(folder, Nc):
    tv = np.asarray(np.load(f'{folder}/target_data.npy', allow_pickle=True).item()['target_gkp_cavity']).ravel()
    v = np.zeros(Nc, complex); v[:min(Nc, tv.size)] = tv[:min(Nc, tv.size)]
    return v/np.linalg.norm(v)


def best_rotation_fidelity(kg, tb, nrot=721):
    """Post-selected fidelity of the (unnormalized) herald state kg, maximized over exp(i th n)."""
    th = np.linspace(0, TWO_PI, nrot)
    rots = np.exp(1j*np.outer(th, np.arange(len(tb))))*tb
    ov = np.abs(rots@np.conj(kg))**2; i = int(np.argmax(ov)); pg = float(np.vdot(kg, kg).real)
    return float(ov[i])/pg, pg, float(np.degrees(th[i]))


def simulate(S, Nc=40, stark='td', scale=1.0, closing='pulse', n_theta=37, n_phi=72):
    """Closed-system displaced-frame propagation of the whole sequence.
    H' = chi/2 sz (n + beta* a + beta a^dag + |beta|^2) + Omega_R/2 r(t) sx + ff(t) sz + H_pi/2(t).
    closing='pulse': the closing pi/2 as played (finite, sign pi2_close_sign);
    closing='best' : instantaneous closing rotation with angle and axis maximizing p_g F,
                     as calibrated on hardware (approx_vs_alpha.py)."""
    import qutip as qt
    chi, tg = S['chi'], S['tg']; ff = stark_ff(S, stark, scale)
    nend = S['NT'] if closing == 'pulse' else S['idx']['end'] + 1
    tl = tg[:nend]
    a = qt.tensor(qt.destroy(Nc), qt.qeye(2)); n = a.dag()*a
    sx = qt.tensor(qt.qeye(Nc), qt.sigmax()); sy = qt.tensor(qt.qeye(Nc), qt.sigmay())
    sz = qt.tensor(qt.qeye(Nc), qt.sigmaz())
    K = lambda v, o=3: qt.coefficient(np.ascontiguousarray(v[:nend]), tlist=tl, order=o)
    be = S['beta']; Om2 = S['Om2']
    pi2 = -Om2/2*S['p2o'] + (S['pi2c_sign']*Om2/2*S['p2c'] if closing == 'pulse' else 0)
    H = [chi/2*sz*n, [sz*a, K(chi/2*np.conj(be))], [sz*a.dag(), K(chi/2*be)],
         [sz, K(chi/2*np.abs(be)**2 + ff)], [sx, K(S['Om']/2*S['er'])], [sy, K(pi2, 1)]]
    t0 = time.time()
    r = qt.sesolve(H, qt.tensor(qt.basis(Nc, 0), qt.basis(2, 0)), tl, options=SIM_OPTS)
    psi = r.final_state.full().reshape(Nc, 2)
    tb = _target_at(S['folder'], Nc); bend = complex(S['beta'][nend - 1])
    D = displacement(bend, Nc)                                   # back to the lab frame
    if closing == 'pulse':
        kg = D@psi[:, 0]
        F, pg, frame = best_rotation_fidelity(kg, tb)
        close = (np.pi/2, 0.0)
    else:
        sy_, sx_ = np.array([[0, -1j], [1j, 0]]), np.array([[0, 1], [1, 0]]); best = (-1,)
        for thc in np.linspace(0, np.pi, n_theta):
            for phc in np.linspace(0, TWO_PI, n_phi, endpoint=False):
                U = np.cos(thc/2)*np.eye(2) - 1j*np.sin(thc/2)*(np.cos(phc)*sy_ + np.sin(phc)*sx_)
                kx = D@(psi@U.T)[:, 0]
                Fx, pgx, fr = best_rotation_fidelity(kx, tb, nrot=361)
                if pgx*Fx > best[0]:
                    best = (pgx*Fx, Fx, pgx, fr, thc, phc, kx)
        _, F, pg, frame, thc, phc, kg = best
        close = (thc, phc)
    return dict(F=F, p_g=pg, F_joint=F*pg, frame_deg=frame % 180, frame_full_deg=frame, beta_end=abs(bend),
                trunc=float(np.abs(psi[-6:, :]).max()), state=kg/np.linalg.norm(kg), close=close,
                stark=(stark, scale), seconds=time.time() - t0)


def schedule_search(folder, Nc=40, k_sb=(3, 4, 5), k_rr=(1, 2), k_pre=(0,), k_dn=(2, 3, 4, 5),
                    t_pi2=84, stark='td', scale=1.0, closing='pulse', key='F',
                    f_rabi_Hz=DEVICE['f_rabi_Hz'], verbose=True):
    """Integer-time search: every segment is k beat intervals rounded to the 4 ns grid
    (beat_aligned(k)); the full sequence is simulated for each combination. closing='best' uses a
    coarse closing-rotation scan (19 x 36)."""
    T_oc = int(np.load(f'{folder}/hardware_params.npy', allow_pickle=True).item()['T_oc_ns'])
    rows = []; t0 = time.time()
    for ks, kr, kp, kd in itertools.product(k_sb, k_rr, k_pre, k_dn):
        sched = (beat_aligned(ks, f_rabi_Hz), t_pi2, beat_aligned(kr, f_rabi_Hz),
                 beat_aligned(kp, f_rabi_Hz), beat_aligned(kd, f_rabi_Hz))
        R = simulate(build_sequence(folder, sched), Nc, stark, scale, closing, n_theta=19, n_phi=36)
        rows.append(dict(sched=sched, k=(ks, kr, kp, kd), F=R['F'], p_g=R['p_g'], F_joint=R['F_joint'],
                         beta_end=R['beta_end'], total_ns=sum(sched) + T_oc + t_pi2))
    rows.sort(key=lambda r: -r[key])
    if verbose:
        print('%d schedules in %.0f s (Nc=%d, Stark %s x%.2g, closing %s), sorted by %s:' % (
            len(rows), time.time() - t0, Nc, stark, scale, closing, key))
        print('  %5s %5s %5s %5s %5s | %7s %7s %7s %8s %7s' % (
            't_sb', 'pi2', 't_rr', 't_pre', 't_dn', 'F', 'p_g', 'p_g*F', '|b_end|', 'total'))
        for r in rows[:min(8, len(rows))]:
            print('  %5d %5d %5d %5d %5d | %7.4f %7.4f %7.4f %8.3f %7.3f' % (
                *r['sched'], r['F'], r['p_g'], r['F_joint'], r['beta_end'], r['total_ns']/1e3))
    return rows[0], rows


# ============================================================================ plots
def plot_convergence(hist, ax=None, label=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(1 - np.asarray(hist['F_joint_max']), lw=1, label=(label or '') + ' max over batch')
    ax.semilogy(1 - np.asarray(hist['F_joint']), lw=1, alpha=.7, label=(label or '') + ' best-objective member')
    ax.set_xlabel('iteration'); ax.set_ylabel(r'$1-F_{\rm joint}$'); ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_charfunc_pair(state, target, xs=np.linspace(-4, 4, 81), titles=('target', 'prepared'), fig_title=''):
    import matplotlib.pyplot as plt
    Ct, Cf = charfunc_grid(target, xs), charfunc_grid(state, xs)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), constrained_layout=True)
    ext = [xs[0], xs[-1], xs[0], xs[-1]]
    for a_, Z, t in zip(ax, (Ct, Cf), titles):
        im = a_.imshow(Z, origin='lower', extent=ext, cmap='RdBu_r', vmin=-1, vmax=1); a_.set_title(t)
    m = max(np.abs(Cf - Ct).max(), 1e-9)
    im2 = ax[2].imshow(Cf - Ct, origin='lower', extent=ext, cmap='PuOr_r', vmin=-m, vmax=m)
    ax[2].set_title('difference (max %.3f)' % m)
    for a_ in ax:
        a_.set_xlabel(r'Re $\beta$')
    ax[0].set_ylabel(r'Im $\beta$')
    fig.colorbar(im, ax=ax[:2], shrink=.8, label=r'Re $C(\beta)$'); fig.colorbar(im2, ax=ax[2], shrink=.8)
    fig.suptitle(fig_title, fontsize=10)
    return fig
