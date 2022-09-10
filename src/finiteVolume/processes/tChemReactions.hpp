#ifndef ABLATELIBRARY_TCHEMREACTIONS_HPP
#define ABLATELIBRARY_TCHEMREACTIONS_HPP

#include <eos/tChem.hpp>
#include "process.hpp"

namespace tChemLib = TChem;

namespace ablate::finiteVolume::processes {

class TChemReactions : public Process {
   private:
    // store some default values
    double dtMin = 1.0E-12;
    double dtMax = 1.0E-1;
    double dtDefault = 1E-4;
    double dtEstimateFactor = 1.5;
    double relToleranceTime = 1.0E-4;
    double absToleranceTime = 1.0E-8;
    double relToleranceNewton = 1.0E-6;
    double absToleranceNewton = 1.0E-10;

    int maxNumNewtonIterations = 100;
    int numTimeIterationsPerInterval = 100000;
    int jacobianInterval = 1;
    int maxAttempts = 4;

    // eos of state variables
    std::shared_ptr<eos::TChem> eos;
    const size_t numberSpecies;

    // tchem memory storage on host/device.  These will be sized for the number of active nodes in the domain
    real_type_2d_view stateDevice;
    real_type_2d_view_host stateHost;

    // store the end state for the device/host
    real_type_2d_view endStateDevice;

    // the time advance information
    time_advance_type_1d_view timeAdvanceDevice;
    time_advance_type timeAdvanceDefault{};

    // store host/device memory for computing state
    real_type_1d_view internalEnergyRefDevice;
    real_type_1d_view_host internalEnergyRefHost;
    real_type_2d_view perSpeciesScratchDevice;

    // store the source terms (density* energy + density*species)
    real_type_2d_view_host sourceTermsHost;
    real_type_2d_view sourceTermsDevice;

    // tolerance constraints
    real_type_2d_view tolTimeDevice;
    real_type_1d_view tolNewtonDevice;
    real_type_2d_view facDevice;

    // store the time and delta for the ode solver
    real_type_1d_view timeViewDevice;
    real_type_1d_view dtViewDevice;

    // store device specific kineticModelGasConstants
    tChemLib::KineticModelConstData<typename Tines::UseThisDevice<exec_space>::type> kineticModelGasConstDataDevice;
    kmd_type_1d_view_host kineticModelDataClone;
    Kokkos::View<KineticModelGasConstData<typename Tines::UseThisDevice<exec_space>::type> *, typename Tines::UseThisDevice<exec_space>::type> kineticModelGasConstDataDevices;

    /**
     * private function to compute the energy and densityYi source terms over the next dt
     * @param flowTs
     * @param flow
     * @return
     */
    PetscErrorCode ChemistryFlowPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);

    static PetscErrorCode AddChemistrySourceToFlow(const FiniteVolumeSolver &solver, DM dm, PetscReal time, Vec locX, Vec fVec, void *ctx);

   public:
    explicit TChemReactions(const std::shared_ptr<eos::EOS> &eos, const std::shared_ptr<ablate::parameters::Parameters> &options = {});
    ~TChemReactions() override;
    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * compute/setup memory for the current mesh
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

    /**
     * public function to copy the source terms to a locFVec
     * @param solver
     * @param fVec
     */
    void AddChemistrySourceToFlow(const FiniteVolumeSolver &solver, Vec locFVec);
};
}  // namespace ablate::finiteVolume::processes
#endif
