#ifndef ABLATELIBRARY_RADIATION_HPP
#define ABLATELIBRARY_RADIATION_HPP

#include <memory>
#include <set>
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "monitors/logs/log.hpp"
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::radiation {

class Radiation : public solver::CellSolver, public solver::RHSFunction {  // Cell solver provides cell based functionality, right hand side function compatibility with finite element/ volume
   public:
    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param rayNumber
     * @param options other options
     */
    Radiation(std::string solverId, std::shared_ptr<domain::Region> region, PetscInt raynumber, std::shared_ptr<parameters::Parameters> options,
              std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel, std::shared_ptr<ablate::monitors::logs::Log> = {});

    /** Returns the black body intensity for a given temperature and emissivity*/
    static PetscReal FlameIntensity(PetscReal epsilon, PetscReal temperature);

    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     *
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) override;

   protected:
    DM radDM; //!< DM associated with the radiation particles

   private:
    /// Class Methods
    void RayInit();

    eos::ThermodynamicTemperatureFunction absorptivityFunction;
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

    /// Class Constants
    const PetscReal sbc = 5.6696e-8;  // Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;

    /// Class inputs and Variables
    PetscInt dim = 0;  // Number of dimensions that the domain exists within
                       //    PetscReal h = 0;   // This is the step size which should be set as the minimum cell radius
    PetscInt nTheta;   // The number of angles to solve with, given by user input
    PetscInt nPhi;     // The number of angles to solve with, given by user input (x2)

    /**
     * Store a log used to output the required information
     */
    const std::shared_ptr<ablate::monitors::logs::Log> log;

    std::vector<std::vector<std::vector<std::vector<std::vector<PetscInt>>>>> rays;  //!< Indices: Cell, angle (theta), angle(phi), space steps (Storing indices at locations)
    std::vector<std::vector<std::vector<std::vector<std::vector<PetscInt>>>>> h;     //!< Indices: Cell, angle (theta), angle(phi), space steps (Storing indices at locations)
    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Ij1;               //!< Indices: Cell, angle (theta), angle(phi), domains (Storing final ray intensity of last time step)
    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Ij;                //!< Indices: Cell, angle (theta), angle(phi), domains (Storing final ray intensity of last time step)
    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Izeros;            //!< Indices: Cell, angle (theta), angle(phi), domains (Storing final ray intensity of last time step)

    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Krad;
    std::vector<std::vector<std::vector<std::vector<PetscReal>>>> Kones;

    PetscInt domainPoints = 10;  //!< The number of points in each domain.
};

}  // namespace ablate::radiation
#endif
