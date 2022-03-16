The ABLATE library is built upon a solver paradigm where individual solvers are instantiated and responsible for setting up the domain. The high level view of setting up a problem is:

1. Create a Domain
    - The Domain holds a DMPlex
    - Create the mesh
    - Define any required labels/subDomains
    - Add any required ghost cells/nodes or boundary cells/nodes

2. Initialize any solvers
    - Define if the solver runs on the entire or subsection of the domain
    - Define any fields that must be stored in the domain/dmPlex
    - Integrate with time stepping

3. Create the timeStepper
    - Handles time-stepping
    - Controls monitors
    - Holds the serialization controller

Each solver can function independently but must support integrating in the same domain. At a minimum solvers must implement the following:

- Define a region in which they apply through Region Label
- Register any required fields with the Domain ```void Register(std::shared_ptr<ablate::domain::SubDomain> subDomain)```
- Setup any functions and time stepping ```void Setup() ```
- Initialize the solution fields ```void Initialize() ```

The three basic types of solvers that are currently supported are:

- finiteVolume: A finite volume implementation with support of computed fluxes between cells
- finiteElement: A finite element implementation with current support for lowMach and incompressible flows
- particles: Support for Lagrangian particles that interact with an Eulerian solver
