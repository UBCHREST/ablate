#ifndef ABLATELIBRARY_EVTRANSPORT_HPP
#define ABLATELIBRARY_EVTRANSPORT_HPP

#include <eos/transport/transportModel.hpp>
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {
/**
 * This class is used to transport any arbitrary extra variable (EV) with a given diffusion coefficient.
 * The variable are assumed to be stored in a conserved form (densityEV) in the solution vector and a
 * non conserved form (EV) in the aux vector.
 */

class EVTransport : public FlowProcess {
   private:
    // store the conserved and non conserved form of the ev.
    const std::string conserved;
    const std::string nonConserved;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;

    // Store ctx needed for static function advection function passed to PETSc
    struct AdvectionData {
        /* number of extra species */
        PetscInt numberEV;

        // EOS function calls
        eos::ThermodynamicFunction computeTemperature;
        eos::ThermodynamicTemperatureFunction computeInternalEnergy;
        eos::ThermodynamicTemperatureFunction computeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction computePressure;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;
    };
    AdvectionData advectionData;

    struct DiffusionData {
        /* number of extra species */
        PetscInt numberEV;

        /* functions to compute diffusion */
        eos::ThermodynamicFunction diffFunction;

        /* store a scratch space for speciesSpeciesSensibleEnthalpy */
        std::vector<PetscReal> speciesSpeciesSensibleEnthalpy;
    };
    DiffusionData diffusionData;

    // Store ctx needed for static function diffusion function passed to PETSc
    PetscInt numberEV;

   public:
    explicit EVTransport(std::string conserved, std::string nonConserved, std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {},
                         std::shared_ptr<eos::transport::TransportModel> transportModel = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * Function to compute the EV fraction. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateEVField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, const PetscInt* aOff,
                                        PetscScalar* auxField, void* ctx);

   private:
    /**
     * This computes the species transfer for species diffusion fluxy
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEVFlux(PetscInt dim, const PetscReal* area, const PetscReal* normal, const PetscReal* centroid, const PetscInt uOff[], const PetscInt uOff_x[],
                                          const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                          PetscScalar flux[], void* ctx);

    /**
     * This Computes the advection flux for each species (Yi)
     * u = {"euler", "densityYi"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar* flux, void* ctx);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SPECIESTRANSPORT_HPP
