#ifndef ABLATELIBRARY_EULERADVECTION_HPP
#define ABLATELIBRARY_EULERADVECTION_HPP

#include <petsc.h>
#include <flow/fluxDifferencer/fluxDifferencer.hpp>
#include "flowProcess.hpp"

namespace ablate::flow::processes {

class EulerAdvection : public FlowProcess {
   public:
    typedef enum { RHO, RHOE, RHOU, RHOV, RHOW } Components;

    struct _EulerAdvectionData {
        /* number of gas species */
        PetscInt numberSpecies;
        PetscReal cfl;

        /* store method used for flux differencer */
        ablate::flow::fluxDifferencer::FluxDifferencerFunction fluxDifferencer;
        void* fluxDifferencerCtx;
        
        // EOS function calls
        PetscErrorCode (*decodeStateFunction)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal* densityYi, PetscReal* internalEnergy, PetscReal* a,
                                              PetscReal* p, void* ctx);
        void* decodeStateFunctionContext;
    };
    typedef struct _EulerAdvectionData* EulerAdvectionData;

    /**
     * public constructor for euler advection
     */
    EulerAdvection(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencerIn = {});

    ~EulerAdvection() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::flow::FVFlow& flow) override;

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
    std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencer;

    // static function to compute time step for euler advection
    static double ComputeTimeStep(TS ts, ablate::flow::Flow& flow, void* ctx);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_EULERADVECTION_HPP
