#ifndef ABLATELIBRARY_EVTRANSPORT_HPP
#define ABLATELIBRARY_EVTRANSPORT_HPP

#include <finiteVolume/fluxCalculator/fluxCalculator.hpp>
#include "process.hpp"

namespace ablate::finiteVolume::processes {

/**
 * This class is used to transport any arbitrary extra variable (EV) with a given diffusion coefficient.
 * The variable are assumed to be stored in a conserved form (densityEV) in the solution vector and a
 * non conserved form (EV) in the aux vector.
 */
class EVTransport : public Process  {
   private:
    // Store the number of variables in this component
    PetscInt numberEV;

    // Store the conserved and non conserved variable names
    const std::string conserved;
    const std::string nonConserved;


    ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
    void* fluxCalculatorCtx;

    /**
     * Update the non conserved form of the eV from the fields {"euler", "density*EV*"}
     */
    static PetscErrorCode UpdateNonConservedEV(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

    static PetscErrorCode EVFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                      const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                      const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);

    /**
     * Function to get the density, velocity conserved variables
     * @return
         */
    static void DecodeEulerState(PetscInt dim, const PetscReal* conservedValues, const PetscReal* normal, PetscReal* density, PetscReal* normalVelocity, PetscReal* velocity);

   public:
    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolume& flow) override;

};

}
#endif  // ABLATELIBRARY_EVTRANSPORT_HPP
