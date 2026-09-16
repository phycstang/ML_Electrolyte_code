# Next stage: reaction-aware reconstructability

## Why this stage is now necessary

The static-parent analysis should be treated as a structural prior, not as a complete predictor. Three observations force this transition:

1. a GaCl3-like local motif can retrospectively recover InI3/AlBr3/ZnCl2, but a single motif does not generalize well across all seven known positives;
2. a richer positive-only multi-prototype dictionary improves internal LOPO coverage but does not preserve the InI3/ZnCl2 blind recovery;
3. exact-6/sixfold halogen order is common in the unlabeled binary-halide background and is not significantly enriched by itself.

The missing variable is therefore not another static-CIF descriptor. It is the **response of the parent structure to the actual chemistry that creates the viscoelastic mixed-anion material**.

The working physical quantity is **reconstructability**: how easily a binary halide can exchange halogen for O, access several nearby mixed-anion configurations, change coordination/connectivity, and soften without simply decomposing into stable crystalline products.

---

## 1. First-principles reaction descriptor

For a generic parent `M_p X_q`, define a charge-balanced O-for-2X exchange against Li2O/LiX:

\[
\Delta E_{O/X} = E(M_pX_{q-2}O) + 2E(LiX)
                 - E(M_pX_q) - E(Li_2O).
\]

For Na chemistry, replace Li2O/LiX by Na2O/NaX.

This is not by itself a glass-formability descriptor. It answers the first question: **is partial anion exchange chemically accessible?**

Important interpretation:

- very positive `DeltaE_OX`: parent resists O incorporation;
- moderately negative/small `DeltaE_OX`: partial exchange may be accessible;
- extremely favorable exchange plus strongly stable crystalline products can instead drive complete reaction/decomposition, as in the general problem highlighted by salt-mixture studies.

Therefore `DeltaE_OX` must be paired with the descriptors below.

---

## 2. Mixed-anion configurational degeneracy

For each selected parent, enumerate symmetry-distinct O substitutions at low concentration and relax them independently.

Recommended first level:

- one O replacing two X-equivalent charge units where chemically meaningful;
- all symmetry-distinct substitution environments in the smallest manageable supercell;
- then a second concentration for the most promising materials.

From the relaxed configurations obtain:

\[
\Delta E_i = E_i - E_{min}
\]

and count

- `N_sub_25`: number of configurations within 25 meV/atom of the minimum;
- `N_sub_50`: number within 50 meV/atom;
- energy spread `sigma_E_sub`;
- an effective configurational entropy / participation measure.

A parent with many structurally distinct low-energy O-substituted states is a more plausible source of crystallization frustration than a parent with one overwhelmingly preferred ordered product.

---

## 3. Structural reconstructability after substitution

For every relaxed substituted structure compare before/after local topology.

Minimum outputs:

- `RMSD_relax`: displacement after removing rigid translation and matching periodic sites;
- `Delta_CN_M`: change in metal coordination number;
- `Delta_CN_XO`: change in anion coordination distribution;
- `Delta_z_poly`: change in number of connected neighboring coordination polyhedra;
- corner/edge/face-sharing changes;
- bond-length coefficient of variation before/after;
- local volume / free-volume change;
- dimensionality change of the M-(X,O) network.

A useful reconstructability descriptor should reward **accessible topology change without catastrophic decomposition**.

---

## 4. Mechanical / vibrational softness

For the most promising mixed-anion structures calculate at least one direct softness measure.

Preferred order of cost:

1. finite-strain elastic tensor / shear moduli;
2. minimum shear eigenvalue or `G_min` if directional elastic response is resolved;
3. low-frequency Gamma modes;
4. small-q phonons for only the final shortlist.

Candidate descriptors:

- Voigt/Reuss/Hill shear modulus `G`;
- elastic anisotropy `A_U`;
- minimum directional shear stiffness;
- integrated low-frequency phonon weight;
- participation of bridging X/O atoms in soft modes.

The goal is not to search for imaginary phonons. It is to quantify low-energy bending/rotation/reconstruction channels.

---

## 5. Phase-selectivity / decomposition control

Reaction thermodynamics must distinguish a useful partially reconstructed mixed-anion state from complete conversion to stable crystalline products.

For each shortlisted chemistry compare:

- parent + Li2O (or Na2O);
- target partially exchanged mixed-anion configurations;
- obvious binary/ternary competing products;
- convex-hull / reaction-energy distance where reliable reference energies are available.

Define a qualitative window:

`reaction accessible` + `many nearby mixed-anion states` + `no overwhelmingly dominant crystalline sink`.

This window is physically closer to viscoelastic precursor formation than the sign of one reaction energy alone.

---

## 6. Experimental / computational study design

### Calibration set

Use all known positives, not just the GaCl3-like subgroup:

- AlCl3
- FeCl3
- GaF3
- InBr3
- TaCl5
- ZrCl4
- GaCl3

Keep InI3, AlBr3, ZnCl2, SnCl2 untouched as retrospective validation until the reaction-aware descriptor definitions are frozen.

### Static-structure controls

Choose several materials that rank high structurally but are not known positives. They are essential controls because they answer whether reaction-aware quantities remove static-structure false positives.

Select controls from **different static views**, for example:

- high local-motif / high REMatch agreement;
- high local-motif / low REMatch disagreement;
- high multi-prototype but moderate single-motif score;
- high exact-6 but unremarkable SOAP score.

Do not choose the controls by chemistry intuition after seeing DFT results.

### Recommended first DFT batch size

A practical first stage is approximately 12-18 parent chemistries:

- 7 known positives;
- 5-8 structurally high-ranked unlabeled controls spanning the categories above;
- optionally 2-3 deliberately low-ranked controls.

Freeze the material list before computing reaction-aware quantities.

---

## 7. Final modeling target

The next model should not be a large supervised classifier. With this sample size, the scientifically meaningful target is an interpretable two-layer ranking:

\[
S_{final} = f(S_{structure}, S_{reconstructability})
\]

where

- `S_structure` summarizes frozen static-parent evidence;
- `S_reconstructability` summarizes reaction accessibility, mixed-anion degeneracy, topology change, and softness.

Start with transparent Pareto analysis or a rank aggregation rather than fitting many free weights.

A material is most compelling when it is simultaneously:

1. close to one of the validated structural families;
2. able to access multiple partially exchanged states;
3. able to reconstruct coordination/connectivity at low energetic cost;
4. mechanically/vibrationally soft enough to support local rearrangement;
5. not dominated by an obvious crystalline decomposition sink.

---

## 8. Paper-level hypothesis after the static study

The structural analysis now motivates a sharper hypothesis:

> Binary metal-halide parent structures do not encode viscoelasticity through one universal triangular or crystalline motif. Instead, they provide several local structural pathways whose usefulness depends on whether reaction with the oxide/alkali-halide chemistry can reconstruct those motifs into a frustrated, dynamically soft mixed-anion network.

This hypothesis is testable. The reaction-aware DFT stage should be considered successful only if it improves separation of known positives from structurally similar controls **without changing descriptor definitions after revealing the blind InI3/AlBr3/ZnCl2/SnCl2 set**.
