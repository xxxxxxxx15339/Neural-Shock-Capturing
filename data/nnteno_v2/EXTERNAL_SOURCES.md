# External dataset search: fidelity and suitability register

Search performed 2026-10-15. Links verified via public search/page results. **No external values have been ingested or redistributed** in the HDF5 corpus. In the current execution environment, direct downloading of some remote archives failed; this status is not disguised as successful collection.

## Actual laboratory observations (physical sensors/cameras)

1. **National Physical Laboratory shock tube calibration (Downes, Knott, Robinson, 2014)**. Original paper DOI: https://doi.org/10.1098/rsta.2013.0299 ; article https://pmc.ncbi.nlm.nih.gov/articles/PMC4115465/. Supplement identified as `rsta20130299supp1.zip` (17.8 MB), data for figures. **Experimental:** pressure sensor response curves, apparatus-specific measurements. **Appropriate use:** independent apparatus validation, subject to sensor calibration and complete metadata. **Not usable without adaptation** as full `(rho,rho*u,E)` training reference.

2. **Henriksen & Bjerketvedt (2024), Experimental data of gas explosions in a 1-meter open-ended channel.** https://usn.figshare.com/articles/dataset/Experimental_data_of_gas_explosions_in_a_1-meter_open-ended_channel/13796108 . CC0; individual CSVs contain four pressure readings, flame-front positions and flame velocity, alongside fuel composition. **Experimental**, but reactive explosion dynamics, not ideal inviscid Euler. Good physical-transfer stress test only with reactive-physics modelling. Collection as a whole is approximately 4.2 GB; select an individual well-documented CSV if needed.

3. **RWTH Aachen low-cost C2H2/O2 combustion-driver shock-tube dataset (2026)** DOI https://doi.org/10.18154/RWTH-2025-02558 ; https://publications.rwth-aachen.de/record/1006589 . Measured voltage recordings of pressure sensors, with a guide and a data ZIP. **Experimental**, chemistry/instrument dynamics require calibration. Do not convert the voltages to pressure without documented sensor response and scaling.

4. **TU Delft / 4TU ResearchData dense-vapor rarefaction shocks (2026)** DOI https://doi.org/10.4121/E29D584F-601A-43F3-A1C6-32FFE5770436 ; https://research.tudelft.nl/en/datasets/data-supporting-the-research-on-generation-and-measurement-of-rar/ . Measured pressure and temperature of siloxane D6 Ludwig-tube experiments. **Experimental** but non-ideal dense-vapor equation of state. **Do not mix into gamma=1.4 ideal-gas Euler reference labels.**

5. **ENDGAME 2024 shock-tube imaging (INGV / Zenodo)** https://zenodo.org/records/13359364 . High-speed Schlieren/shadow imaging at 30,000 frames/s and vent pressure measurements; some variants contain fluid/particles. **Experimental**, valuable visualization/qualitative shock motion, but images are not cell-averaged conservative-variable fields.

## Published numerical simulation datasets (NOT physical observations)

6. **PDEBench** DOI https://doi.org/10.18419/DARUS-2986 ; https://github.com/pdebench/PDEBench ; raw https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/DARUS-2986 . HDF5 simulation trajectories, common convention `[batch,time,spatial,...,variables]`. Includes advection, Burgers and compressible Navier–Stokes. The 1D CFD archive is **not generally inviscid Euler**; inspect viscosity (`eta`,`zeta`), equation parameters, boundary conditions and conservative/primitive variable definitions before use. External held-out/out-of-model-distribution test, not automatically compatible ground truth.

7. **Clawpack Riemann book** https://github.com/clawpack/riemann_book/tree/master/exact_solvers ; https://www.clawpack.org/gallery/pyclaw/gallery/shocktube.html . Verified classical exact ideal-gas Euler and scalar Riemann solvers and high-resolution approximation infrastructure. Suitable as independent cross-check for *mathematical* PDE reference values. This source is software, not recorded experimental measurements.

8. **Guermond et al. high-order shock-tube simulation archive** DOI https://doi.org/10.5281/zenodo.4895237 . High-order Navier–Stokes simulation of a *viscous* shock tube, with tabulated `x-rho-p` fields at grid levels. Could support a specialized fluid code cross-check; not ideal-gas inviscid 1D Euler and not laboratory data.

## Quality threshold for an experimental comparison

- Original raw recording; sensor serial/calibration, units, uncertainty and acquisition rate.
- Apparatus geometry, wall conditions, upstream/downstream gas properties and time-zero trigger.
- Information needed to map physical measurements to simulated quantities at the **same sensor location** and sample time.
- Conservation-law/model discrepancy analysis (viscosity, chemistry, heat transfer, 2D, reflections, measurement response).
- Entire experimental run retained within a single split; zero row/window or repeated-experiment leakage.

**No physical measurements are in the current archive because they are not commensurate with the required full-state supervised targets without experiment-specific modelling.** Fabricating unmeasured momentum/energy would violate the user's quality requirement.
