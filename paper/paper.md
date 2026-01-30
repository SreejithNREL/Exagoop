---
title: "ExaGOOP: an AMReX-based material point method solver"
tags:
  - C++
  - material point method
  - Exascale
  - heterogenous computing
authors:
  - name: Sreejith N. A.
    orcid: 0000-0001-5685-4070
    affiliation: 1
  - name: Nicholas Deak
    orcid: 0000-0003-1543-9334
    affiliation: 1
  - name: Yudong Li
    orcid: 0000-0002-7024-433X
    affiliation: 2
  - name: Hariswaran Sitaraman
    orcid: 0000-0001-5304-1664
    affiliation: 1
  - name: Marc Day
    orcid: 0000-0002-1711-3963
    affiliation: 1
affiliations:
  - name: Scalable Algorithms, Modeling and Simulation (SAMS) Group, National Renewable Energy Laboratory, USA
    index: 1
  - name: Catalytic Carbon Transformation and Scale-Up Center, National Renewable Energy Laboratory, USA
    index: 2  
date: 30 August 2025
bibliography: paper.bib
---

# Summary

ExaGOOP is a versatile continuum mechanics solver based on the material point method (MPM). Traditional numerical solvers for continuum physics typically employ finite difference,volume, or element methods. These approaches require the entire computational domain to be discretized using a computational grid, where the governing equations are solved in differential, integral or variational forms. The necessity of a computational grid with complex geometries makes it challenging to solve problems involving highly deformable and history-dependent materials and multiphase systems. In contrast, MPM, like many particle-based methods, is based on a Lagrangian formulation of the governing equations. Unlike conventional solvers that rely on grids, MPM stores the material's properties on a collection of particles (also called as material points). While MPM does require a background mesh, it is most often a uniform Cartesian grid, and used only as a temporary construct for calculating gradients and is reset after each time integration step. This approach effectively eliminates issues associated with grid element deformations. As a result, MPM methods are particularly well-suited for a wide range of continuum mechanics problems, especially those that involve significant material deformations.

ExaGOOP leverages the AMReX [@zhang2019amrex] library, which has been widely utilized in adaptive Cartesian grid and particle based applications [@PeleLMeX_JOSS; @PeleSoftware; @Sitaraman2021; @deak2025high]. The AMReX library facilitates the generation of a block-structured, Cartesian background grid within ExaGOOP and efficient parallelization using distributed memory and performance portable shared memory paradigms. The material points related operations are managed by the particle classes provided by AMReX. 
 Currently, the implementation supports a uniform grid without refinement, but the extension to adaptive grids is part of our future efforts. 

The various steps in an MPM time update include particle-to-grid (P2G), nodal velocity update, grid-to-particle (G2P), and particle position update. ExaGOOP offers users the flexibility to select the spatial discretization scheme, allowing for the use of linear-hat, quadratic B-spline, or cubic B-spline shape functions for both the P2G and G2P operations. Currently, the nodal update is performed using explicit Euler time integration; however, implicit time stepping schemes are part of ongoing work and is present in beta testing branches. In addition to these options, ExaGOOP allows users to select various numerical input parameters, such as the particle-in-cell (PIC)-Fluid Implicit Particle (FIP) blending factor in the G2P step, and whether to use Update Stress Last (USL) or Modified Update Stress Last (MUSL) for stress calculations. The solver also supports CFL-based adaptive time-stepping. At present, ExaGOOP supports barotropic fluid and linear elastic solid constitutive models. However, adding new constitutive models is relatively straightforward for users, requiring only the development of the new constitutive model function without necessitating changes to other parts of the code. Complex, static wall boundaries are simulated using the level set method, while moving boundaries can be simulated with fictitious rigid material points.

ExaGOOP has undergone extensive validation and verification using 1D, 2D, and 3D test cases, all of which are available in the GitHub repository. Preprocessing scripts in the repository enable users to generate initial material point distribution with the desired number of material points per cell for either for user-defined simple geometries or based on user-provided images of complex bodies. Users can specify the constitutive model for each material point, facilitating multi-body and multi-phase simulations with ease.

ExaGOOP was developed and is actively maintained in C++ and utilizes parallelization subroutines from the AMReX library. It employs an MPI+X approach, where Message Passing Interface (MPI) is used to distribute Cartesian grid patches and co-located particles across different distributed memory ranks. Each grid can be further divided into logical tiles, which can be distributed among threads using shared-memory OpenMP on multi-core Central-processing-units (CPU) based machines or among Graphics-processing-units (GPU) threads on NVIDIA/AMD/Intel based GPU-accelerated systems.

# Statement of Need

There are numerous MPM solvers available online, such as Karamelo [@devaucorbeil2021karamelo], Matter [@blatny2025matter], GEOS-MPM [@kumar2019geosmpm], and Taichi-MPM [@hu2018taichimpm], but what truly distinguishes ExaGOOP is its performance portability. This allows ExaGOOP to excel on CPU, GPU, and hybrid architectures, making it exceptionally versatile. The advanced memory management, powerful parallel processing capabilities, and robust embedded boundary support offered by the use of AMReX render ExaGOOP as a performance-portable MPM solver. The remarkable exascale performance demonstrated by AMReX in various other solvers highlights ExaGOOP's extraordinary potential to efficiently manage billions of particles on GPU-accelerated and heterogeneous computing systems.

ExaGOOP is intended for students, researchers, and engineers interested in simulating multi-material dynamics involving severe deformations. Originally developed as a tool for studying membrane compaction in high-pressure reverse osmosis applications [@nrel2023amrexmpm; @osti_2000945], ExaGOOP is now being used for simulating continumm mechanics in a variety of other applications such as in lithium-ion battery manufacturing and biomass feedstock flows.

# Software design

ExaGOOP is a modular, performance-portable framework for generalized material point method (MPM) simulations. Its architecture is guided by several key objectives: portability across diverse computing systems, support for heterogeneous CPU and GPU architectures, extensibility of solver capabilities, and strict modularity to ensure a user- and developer-friendly environment. To achieve these objectives, ExaGOOP is constructed on the AMReX framework, which offers a mature, exascale-capable infrastructure with established performance portability. The Cartesian mesh operations in AMReX provide a suitable foundation for the MPM background grid, while its native adaptive mesh refinement supports ExaGOOP’s planned adoption of refined background grids. The inclusion of built-in linear solvers facilitates the extension to implicit time-integration schemes. Additionally, embedded boundary (EB) data structures streamline the representation of essential boundary conditions. Furthermore, AMReX’s particle classes correspond directly to material points, enabling ExaGOOP to perform particle–grid transfers and constitutive updates efficiently.

Most existing MPM packages are domain-specific, monolithic, or closely integrated with particular physics models, which limits their adaptability for methodological research. Contributing to these projects would not have aligned with ExaGOOP’s objectives of physics-agnostic design, modular extensibility, and sustained performance portability. Developing ExaGOOP as an independent framework enables researchers to introduce new physics via constitutive models while utilizing AMReX’s established scalability and exascale-capable infrastructure.

# Research impact statement

ExaGOOP has facilitated multiple scientific investigations, demonstrating its utility as a flexible, reproducible MPM research framework. For example, it enabled the study of membrane support-layer compaction under ultra-high-pressure reverse osmosis (UHPRO) conditions, in which material points were directly generated from scanning electron microscopy (SEM) images. This approach allowed for an accurate representation of membrane microstructure with minimal preprocessing. The simulated compaction results closely aligned with experimental data on overall strain and qualitative pore deformation [@osti_2000945]. Additionally, ExaGOOP supported a numerical study on the stability characteristics of standard MPM formulations [@osti_2000743], yielding insights into algorithmic behavior that are challenging to obtain with monolithic or application-specific codes. The framework has also been applied to simulations involving lithium-ion battery materials and biomass feedstock mechanics [@osti_2406179], underscoring its versatility across diverse research domains.

The project is designed to promote community adoption and facilitate near-term impact. Its documentation offers a comprehensive introduction to MPM theory, step-by-step tutorials, and validation examples, enabling new users to reproduce results and extend the framework with confidence. The repository features a permissive open-source license, clear contribution guidelines, and versioned releases, which support transparency and long-term maintainability. Collectively, these features establish ExaGOOP as a credible and extensible platform for advancing future MPM research and methodological innovation.

# AI usage disclosure

No generative AI tools were used in the development of this software, the writing of this manuscript, or the preparation of supporting materials.

# Acknowledgements

The development of this software was supported by the National Alliance for Water Innovation (NAWI), funded by the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy (EERE), Advanced Manufacturing Office, under Funding Opportunity Announcement Number DE-FOA-0001905. All of the research was performed using computational resources sponsored by the Department of Energy’s Office of Energy Efficiency and Renewable Energy and located at the National Renewable Energy Laboratory. This work was authored in part by the National Renewable Energy Laboratory, operated by Alliance for Sustainable Energy, LLC, for the U.S. Department of Energy (DOE) under Contract No. DE-AC36-08GO28308. The views expressed in the article do not
necessarily represent the views of the DOE or the U.S. Government. The U.S. Government retains and the publisher, by accepting the article for publication, acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this work, or allow others to do so, for U.S. Government purposes.

# References


