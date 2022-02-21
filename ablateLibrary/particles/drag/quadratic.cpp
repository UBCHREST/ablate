#include "quadratic.hpp"
#include "utilities/mathUtilities.hpp"

void ablate::particles::drag::Quadratic::ComputeDragForce(const PetscInt dim, const PetscReal *partVel, const PetscReal *flowVel, const PetscReal muF, const PetscReal rhoF, const PetscReal partDiam,
                                                          PetscReal *dragForce) {
    /** \brief Quadratic drag formula for a solid sphere at high Reynolds numbers.
     *
     * \details \f$\vec{F}_d = -C_d \frac{\pi}{8} d^2 \cdot \frac{1}{2} \rho_f |\vec{V}| \vec{V}\f$
     *
     * where \f$\vec{V}\f$ is the relative velocity: \f$\vec{V} = \vec{V}_p - \vec{V}_f\f$. The subscript p refers to a particle. The subscript f refers to the ambient fluid.
     *
     * The specific value of \f$C_d\f$ chosen here is the high Reynolds number limit of equation 8 of Loth.
     *
     * References: E. Loth, “Quasi-steady shape and drag of deformable bubbles and drops,” International Journal of Multiphase Flow, vol. 34, no. 6, pp. 523–546, Jun. 2008,
     * doi: 10.1016/j.ijmultiphaseflow.2007.08.010. (See eq. 4.)
     */

    PetscReal relVel[3];
    PetscReal dragForcePrefactor = -0.42 * (PETSC_PI / 8.0) * partDiam * partDiam * rhoF;

    for (int n = 0; n < dim; n++) {
        relVel[n] = partVel[n] - flowVel[n];
    }

    dragForcePrefactor *= ablate::utilities::MathUtilities::MagVector(dim, relVel);

    for (int n = 0; n < dim; n++) {
        dragForce[n] = dragForcePrefactor * relVel[n];
    }
};