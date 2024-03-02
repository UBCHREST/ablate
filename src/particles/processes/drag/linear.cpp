#include "linear.hpp"

void ablate::particles::processes::drag::Linear::ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF,
                                                                  const PetscReal partDiam, PetscReal *dragForce) {
    /** \brief Linear drag formula for a solid sphere at low Reynolds numbers.
     *
     * \details \f$\vec{F}_d = -3 \pi d^2 \mu_f \vec{V}\f$
     *
     * where \f$\vec{V}\f$ is the relative velocity: \f$\vec{V} = \vec{V}_p - \vec{V}_f\f$. The subscript p refers to a particle. The subscript f refers to the ambient fluid.
     *
     * References: E. Loth, “Quasi-steady shape and drag of deformable bubbles and drops,” International Journal of Multiphase Flow, vol. 34, no. 6, pp. 523–546, Jun. 2008,
     * doi: 10.1016/j.ijmultiphaseflow.2007.08.010. (See eq. 3.)
     */

    PetscReal dragForcePrefactor = -3.0 * PETSC_PI * partDiam * muF;

    for (int n = 0; n < dim; n++) {
        dragForce[n] = dragForcePrefactor * (partVel[n] - flowVel[n]);
    }
}

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::particles::processes::drag::DragModel, ablate::particles::processes::drag::Linear, "Computes drag according to Stokes' law.");