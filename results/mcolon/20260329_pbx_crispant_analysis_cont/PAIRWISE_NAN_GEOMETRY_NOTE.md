# Pairwise NaN Geometry Note

The pairwise classification outputs already preserve off-support probe entries as `NaN` in `raw_coordinates` and `shrunk_coordinates`.

The artifact enters downstream when sparse pairwise coordinates are exported or analyzed as if those `NaN` values were neutral `0.0` values. In this representation, `0.0` is a real signed-margin value, not a synonym for "not comparable".

Observed failure mode:
- `inj_ctrl` and `wik_ab` remain mostly near-null on the direct `inj_ctrl__vs__wik_ab` probe.
- They become spuriously separable when represented through external probes such as `inj_ctrl__vs__pbx1b_pbx4_crispant` and `pbx1b_pbx4_crispant__vs__wik_ab`.
- The synthetic split appears when one class gets a real value on one probe and the other class gets `0.0` because that probe was off-support and zero-filled.

Implementation consequence:
- Pairwise geometry must be NaN-aware at the embryo level: compare embryos only on shared valid probes.
- Pairwise null audits must be NaN-aware at the class/bin level: only score probe axes that have support in both classes within the bin.
- Dense zero-filled pairwise vectors are retained only as explicit legacy/debug exports and should not drive the primary cosmology path.
