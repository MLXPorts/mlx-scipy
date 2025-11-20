from typing import Any

import mlx.core as mx

__all__ = [
    'geterr',
    'seterr',
    'errstate',
    'agm',
    'airy',
    'airye',
    'bdtr',
    'bdtrc',
    'bdtri',
    'bdtrik',
    'bdtrin',
    'bei',
    'beip',
    'ber',
    'berp',
    'besselpoly',
    'beta',
    'betainc',
    'betaincc',
    'betainccinv',
    'betaincinv',
    'betaln',
    'binom',
    'boxcox',
    'boxcox1p',
    'btdtria',
    'btdtrib',
    'cbrt',
    'chdtr',
    'chdtrc',
    'chdtri',
    'chdtriv',
    'chndtr',
    'chndtridf',
    'chndtrinc',
    'chndtrix',
    'cosdg',
    'cosm1',
    'cotdg',
    'dawsn',
    'ellipe',
    'ellipeinc',
    'ellipj',
    'ellipk',
    'ellipkinc',
    'ellipkm1',
    'elliprc',
    'elliprd',
    'elliprf',
    'elliprg',
    'elliprj',
    'entr',
    'erf',
    'erfc',
    'erfcinv',
    'erfcx',
    'erfi',
    'erfinv',
    'eval_chebyc',
    'eval_chebys',
    'eval_chebyt',
    'eval_chebyu',
    'eval_gegenbauer',
    'eval_genlaguerre',
    'eval_hermite',
    'eval_hermitenorm',
    'eval_jacobi',
    'eval_laguerre',
    'eval_legendre',
    'eval_sh_chebyt',
    'eval_sh_chebyu',
    'eval_sh_jacobi',
    'eval_sh_legendre',
    'exp1',
    'exp10',
    'exp2',
    'expi',
    'expit',
    'expm1',
    'expn',
    'exprel',
    'fdtr',
    'fdtrc',
    'fdtri',
    'fdtridfd',
    'fresnel',
    'gamma',
    'gammainc',
    'gammaincc',
    'gammainccinv',
    'gammaincinv',
    'gammaln',
    'gammasgn',
    'gdtr',
    'gdtrc',
    'gdtria',
    'gdtrib',
    'gdtrix',
    'hankel1',
    'hankel1e',
    'hankel2',
    'hankel2e',
    'huber',
    'hyp0f1',
    'hyp1f1',
    'hyp2f1',
    'hyperu',
    'i0',
    'i0e',
    'i1',
    'i1e',
    'inv_boxcox',
    'inv_boxcox1p',
    'it2i0k0',
    'it2j0y0',
    'it2struve0',
    'itairy',
    'iti0k0',
    'itj0y0',
    'itmodstruve0',
    'itstruve0',
    'iv',
    'ive',
    'j0',
    'j1',
    'jn',
    'jv',
    'jve',
    'k0',
    'k0e',
    'k1',
    'k1e',
    'kei',
    'keip',
    'kelvin',
    'ker',
    'kerp',
    'kl_div',
    'kn',
    'kolmogi',
    'kolmogorov',
    'kv',
    'kve',
    'log1p',
    'log_expit',
    'log_ndtr',
    'log_wright_bessel',
    'loggamma',
    'logit',
    'lpmv',
    'mathieu_a',
    'mathieu_b',
    'mathieu_cem',
    'mathieu_modcem1',
    'mathieu_modcem2',
    'mathieu_modsem1',
    'mathieu_modsem2',
    'mathieu_sem',
    'modfresnelm',
    'modfresnelp',
    'modstruve',
    'nbdtr',
    'nbdtrc',
    'nbdtri',
    'nbdtrik',
    'nbdtrin',
    'ncfdtr',
    'ncfdtri',
    'ncfdtridfd',
    'ncfdtridfn',
    'ncfdtrinc',
    'nctdtr',
    'nctdtridf',
    'nctdtrinc',
    'nctdtrit',
    'ndtr',
    'ndtri',
    'ndtri_exp',
    'nrdtrimn',
    'nrdtrisd',
    'obl_ang1',
    'obl_ang1_cv',
    'obl_cv',
    'obl_rad1',
    'obl_rad1_cv',
    'obl_rad2',
    'obl_rad2_cv',
    'owens_t',
    'pbdv',
    'pbvv',
    'pbwa',
    'pdtr',
    'pdtrc',
    'pdtri',
    'pdtrik',
    'poch',
    'powm1',
    'pro_ang1',
    'pro_ang1_cv',
    'pro_cv',
    'pro_rad1',
    'pro_rad1_cv',
    'pro_rad2',
    'pro_rad2_cv',
    'pseudo_huber',
    'psi',
    'radian',
    'rel_entr',
    'rgamma',
    'round',
    'shichi',
    'sici',
    'sindg',
    'smirnov',
    'smirnovi',
    'spence',
    'stdtr',
    'stdtridf',
    'stdtrit',
    'struve',
    'tandg',
    'tklmbda',
    'voigt_profile',
    'wofz',
    'wright_bessel',
    'wrightomega',
    'xlog1py',
    'xlogy',
    'y0',
    'y1',
    'yn',
    'yv',
    'yve',
    'zetac'
]

def geterr() -> dict[str, str]: ...
def seterr(**kwargs: str) -> dict[str, str]: ...

class errstate:
    def __init__(self, **kargs: str) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: Any,  # Unused
        exc_value: Any,  # Unused
        traceback: Any,  # Unused
    ) -> None: ...

_cosine_cdf: mx.ufunc
_cosine_invcdf: mx.ufunc
_cospi: mx.ufunc
_ellip_harm: mx.ufunc
_factorial: mx.ufunc
_gen_harmonic: mx.ufunc
_igam_fac: mx.ufunc
_kolmogc: mx.ufunc
_kolmogci: mx.ufunc
_kolmogp: mx.ufunc
_lambertw: mx.ufunc
_lanczos_sum_expg_scaled: mx.ufunc
_lgam1p: mx.ufunc
_log1mexp: mx.ufunc
_log1pmx: mx.ufunc
_normalized_gen_harmonic: mx.ufunc
_riemann_zeta: mx.ufunc
_scaled_exp1: mx.ufunc
_sf_error_test_function: mx.ufunc
_sinpi: mx.ufunc
_smirnovc: mx.ufunc
_smirnovci: mx.ufunc
_smirnovp: mx.ufunc
_spherical_in: mx.ufunc
_spherical_in_d: mx.ufunc
_spherical_jn: mx.ufunc
_spherical_jn_d: mx.ufunc
_spherical_kn: mx.ufunc
_spherical_kn_d: mx.ufunc
_spherical_yn: mx.ufunc
_spherical_yn_d: mx.ufunc
_stirling2_inexact: mx.ufunc
_struve_asymp_large_z: mx.ufunc
_struve_bessel_series: mx.ufunc
_struve_power_series: mx.ufunc
_zeta: mx.ufunc
agm: mx.ufunc
airy: mx.ufunc
airye: mx.ufunc
bdtr: mx.ufunc
bdtrc: mx.ufunc
bdtri: mx.ufunc
bdtrik: mx.ufunc
bdtrin: mx.ufunc
bei: mx.ufunc
beip: mx.ufunc
ber: mx.ufunc
berp: mx.ufunc
besselpoly: mx.ufunc
beta: mx.ufunc
betainc: mx.ufunc
betaincc: mx.ufunc
betainccinv: mx.ufunc
betaincinv: mx.ufunc
betaln: mx.ufunc
binom: mx.ufunc
boxcox1p: mx.ufunc
boxcox: mx.ufunc
btdtria: mx.ufunc
btdtrib: mx.ufunc
cbrt: mx.ufunc
chdtr: mx.ufunc
chdtrc: mx.ufunc
chdtri: mx.ufunc
chdtriv: mx.ufunc
chndtr: mx.ufunc
chndtridf: mx.ufunc
chndtrinc: mx.ufunc
chndtrix: mx.ufunc
cosdg: mx.ufunc
cosm1: mx.ufunc
cotdg: mx.ufunc
dawsn: mx.ufunc
ellipe: mx.ufunc
ellipeinc: mx.ufunc
ellipj: mx.ufunc
ellipk: mx.ufunc
ellipkinc: mx.ufunc
ellipkm1: mx.ufunc
elliprc: mx.ufunc
elliprd: mx.ufunc
elliprf: mx.ufunc
elliprg: mx.ufunc
elliprj: mx.ufunc
entr: mx.ufunc
erf: mx.ufunc
erfc: mx.ufunc
erfcinv: mx.ufunc
erfcx: mx.ufunc
erfi: mx.ufunc
erfinv: mx.ufunc
eval_chebyc: mx.ufunc
eval_chebys: mx.ufunc
eval_chebyt: mx.ufunc
eval_chebyu: mx.ufunc
eval_gegenbauer: mx.ufunc
eval_genlaguerre: mx.ufunc
eval_hermite: mx.ufunc
eval_hermitenorm: mx.ufunc
eval_jacobi: mx.ufunc
eval_laguerre: mx.ufunc
eval_legendre: mx.ufunc
eval_sh_chebyt: mx.ufunc
eval_sh_chebyu: mx.ufunc
eval_sh_jacobi: mx.ufunc
eval_sh_legendre: mx.ufunc
exp10: mx.ufunc
exp1: mx.ufunc
exp2: mx.ufunc
expi: mx.ufunc
expit: mx.ufunc
expm1: mx.ufunc
expn: mx.ufunc
exprel: mx.ufunc
fdtr: mx.ufunc
fdtrc: mx.ufunc
fdtri: mx.ufunc
fdtridfd: mx.ufunc
fresnel: mx.ufunc
gamma: mx.ufunc
gammainc: mx.ufunc
gammaincc: mx.ufunc
gammainccinv: mx.ufunc
gammaincinv: mx.ufunc
gammaln: mx.ufunc
gammasgn: mx.ufunc
gdtr: mx.ufunc
gdtrc: mx.ufunc
gdtria: mx.ufunc
gdtrib: mx.ufunc
gdtrix: mx.ufunc
hankel1: mx.ufunc
hankel1e: mx.ufunc
hankel2: mx.ufunc
hankel2e: mx.ufunc
huber: mx.ufunc
hyp0f1: mx.ufunc
hyp1f1: mx.ufunc
hyp2f1: mx.ufunc
hyperu: mx.ufunc
i0: mx.ufunc
i0e: mx.ufunc
i1: mx.ufunc
i1e: mx.ufunc
inv_boxcox1p: mx.ufunc
inv_boxcox: mx.ufunc
it2i0k0: mx.ufunc
it2j0y0: mx.ufunc
it2struve0: mx.ufunc
itairy: mx.ufunc
iti0k0: mx.ufunc
itj0y0: mx.ufunc
itmodstruve0: mx.ufunc
itstruve0: mx.ufunc
iv: mx.ufunc
ive: mx.ufunc
j0: mx.ufunc
j1: mx.ufunc
jn: mx.ufunc
jv: mx.ufunc
jve: mx.ufunc
k0: mx.ufunc
k0e: mx.ufunc
k1: mx.ufunc
k1e: mx.ufunc
kei: mx.ufunc
keip: mx.ufunc
kelvin: mx.ufunc
ker: mx.ufunc
kerp: mx.ufunc
kl_div: mx.ufunc
kn: mx.ufunc
kolmogi: mx.ufunc
kolmogorov: mx.ufunc
kv: mx.ufunc
kve: mx.ufunc
log1p: mx.ufunc
log_expit: mx.ufunc
log_ndtr: mx.ufunc
log_wright_bessel: mx.ufunc
loggamma: mx.ufunc
logit: mx.ufunc
lpmv: mx.ufunc
mathieu_a: mx.ufunc
mathieu_b: mx.ufunc
mathieu_cem: mx.ufunc
mathieu_modcem1: mx.ufunc
mathieu_modcem2: mx.ufunc
mathieu_modsem1: mx.ufunc
mathieu_modsem2: mx.ufunc
mathieu_sem: mx.ufunc
modfresnelm: mx.ufunc
modfresnelp: mx.ufunc
modstruve: mx.ufunc
nbdtr: mx.ufunc
nbdtrc: mx.ufunc
nbdtri: mx.ufunc
nbdtrik: mx.ufunc
nbdtrin: mx.ufunc
ncfdtr: mx.ufunc
ncfdtri: mx.ufunc
ncfdtridfd: mx.ufunc
ncfdtridfn: mx.ufunc
ncfdtrinc: mx.ufunc
nctdtr: mx.ufunc
nctdtridf: mx.ufunc
nctdtrinc: mx.ufunc
nctdtrit: mx.ufunc
ndtr: mx.ufunc
ndtri: mx.ufunc
ndtri_exp: mx.ufunc
nrdtrimn: mx.ufunc
nrdtrisd: mx.ufunc
obl_ang1: mx.ufunc
obl_ang1_cv: mx.ufunc
obl_cv: mx.ufunc
obl_rad1: mx.ufunc
obl_rad1_cv: mx.ufunc
obl_rad2: mx.ufunc
obl_rad2_cv: mx.ufunc
owens_t: mx.ufunc
pbdv: mx.ufunc
pbvv: mx.ufunc
pbwa: mx.ufunc
pdtr: mx.ufunc
pdtrc: mx.ufunc
pdtri: mx.ufunc
pdtrik: mx.ufunc
poch: mx.ufunc
powm1: mx.ufunc
pro_ang1: mx.ufunc
pro_ang1_cv: mx.ufunc
pro_cv: mx.ufunc
pro_rad1: mx.ufunc
pro_rad1_cv: mx.ufunc
pro_rad2: mx.ufunc
pro_rad2_cv: mx.ufunc
pseudo_huber: mx.ufunc
psi: mx.ufunc
radian: mx.ufunc
rel_entr: mx.ufunc
rgamma: mx.ufunc
round: mx.ufunc
shichi: mx.ufunc
sici: mx.ufunc
sindg: mx.ufunc
smirnov: mx.ufunc
smirnovi: mx.ufunc
spence: mx.ufunc
stdtr: mx.ufunc
stdtridf: mx.ufunc
stdtrit: mx.ufunc
struve: mx.ufunc
tandg: mx.ufunc
tklmbda: mx.ufunc
voigt_profile: mx.ufunc
wofz: mx.ufunc
wright_bessel: mx.ufunc
wrightomega: mx.ufunc
xlog1py: mx.ufunc
xlogy: mx.ufunc
y0: mx.ufunc
y1: mx.ufunc
yn: mx.ufunc
yv: mx.ufunc
yve: mx.ufunc
zetac: mx.ufunc

