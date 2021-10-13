#ifndef ABLATELIBRARY_EULERADVECTION_HPP
#define ABLATELIBRARY_EULERADVECTION_HPP

#include <petsc.h>
#include "flow/fluxCalculator/fluxCalculator.hpp"
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class EulerAdvection : public Process {
   public:
    typedef enum { RHO, RHOE, RHOU, RHOV, RHOW } Components;

    struct _EulerAdvectionData {
        /* number of gas species */
        PetscInt numberSpecies;
        PetscReal cfl;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;

        // EOS function calls
        PetscErrorCode (*decodeStateFunction)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal* densityYi, PetscReal* internalEnergy, PetscReal* a,
                                              PetscReal* p, void* ctx);
        void* decodeStateFunctionContext;
    };
    typedef struct _EulerAdvectionData* EulerAdvectionData;

    /**
     * public constructor for euler advection
     */
    EulerAdvection(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {});

    ~EulerAdvection() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FVFlow& flow) override;

    /**
     * This Computes the Flow Euler flow for rho, rhoE, and rhoVel.
     * u = {"euler"} or {"euler", "densityYi"} if species are tracked
     * a = {}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                           const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                           const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux,
                                                           void* ctx);

    /**
     * This Computes the advection flux for each species (Yi)
     * u = {"euler", "densityYi"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode CompressibleFlowSpeciesAdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                               const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                               const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux,
                                                               void* ctx);

   private:
    EulerAdvectionData eulerAdvectionData;
    std::shared_ptr<eos::EOS> eos;
    std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;

    // static function to compute time step for euler advection
    static double ComputeTimeStep(TS ts, ablate::finiteVolume::Flow& flow, void* ctx);

    /**
     * Private function to decode the euler fields
     * @param flowData
     * @param dim
     * @param conservedValues
     * @param densityYi
     * @param normal
     * @param density
     * @param normalVelocity
     * @param velocity
     * @param internalEnergy
     * @param a
     * @param M
     * @param p
     */
    static void DecodeEulerState(ablate::finiteVolume::processes::EulerAdvection::EulerAdvectionData flowData, PetscInt dim, const PetscReal* conservedValues, const PetscReal* densityYi,
                                 const PetscReal* normal, PetscReal* density, PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_EULERADVECTION_HPP
