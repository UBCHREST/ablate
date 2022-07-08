#ifndef ABLATELIBRARY_RADIATION_HPP
#define ABLATELIBRARY_RADIATION_HPP

#include <memory>
#include <set>
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "monitors/logs/log.hpp"
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/loggable.hpp"

namespace ablate::radiation {

class Radiation : public solver::CellSolver,
                  public solver::RHSFunction,
                  public utilities::Loggable<Radiation> {  //!< Cell solver provides cell based functionality, right hand side function compatibility with
                                                           //!< finite element/ volume, loggable allows for the timing and tracking of events
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
    DM radsolve;   //!< DM associated with the radiation particles
    DM radsearch;  //!< DM which the search particles occupy

   private:
    /// Structs to hold information

    /** Carriers are attached to the solve particles and bring ray information from the local segments to the origin cells
     * They are transported directly from the segment to the origin. They carry only the values that the Segment computes and not the spatial information necessary to  */
    struct Carrier {
        PetscReal Ij = 0;    //!< Black body source for the segment. Make sure that this is reset every solve after the value has been transported.
        PetscReal Krad = 1;  //!< Absorption for the segment. Make sure that this is reset every solve after the value has been transported.
        PetscReal I0 = 0;
    };

    /** Each origin cell will need to retain local information given to it by the ray segments in order to compute the final intenisty.
     * This information will be owned by cell index and stored in a map of local cell indices.
     * */
    struct Origin {
        PetscReal I0 = 0;                   //!< Determing the initial ray intensity by grabbing the head cell of the furthest ray? There will need to be additional setup for this.
        PetscReal Isource = 0;              //!< Value that will be contributed to by every ray segment.
        PetscReal Kradd = 1;                //!< Value that will be contributed to by every ray segment.
        PetscReal intensity = 0;            //!<  Value that will be contributed to by every ray.
        std::map<std::string, Carrier> handler;  //!< Stores local carrier information
        PetscInt nsegmax = 0;               //!< Number of segments that are in this carrier ray. TODO: (Maybe don't need to store this, replace with map function call)
    };

    /** Segments belong to the local maps and hold all of the local information about the ray segments both during the search and the solve */
    struct Segment {
        std::vector<PetscInt> cells;  //!< Stores the cell indices of the segment locally.
        std::vector<PetscReal> h;     //!< Stores the space steps of the segment locally.
    };

    /** Identifiers are carrying by both the search and solve particles in order to associate them with their origins and ray segments
     * In the search particles, nsegment iterates based on how many domains the search particle has crossed.
     * In the solve particle, nsegment remains constant as it ties the particle to its specific order in the ray */
    struct Identifier {
        PetscInt origin;
        PetscInt iCell;
        PetscInt ntheta;
        PetscInt nphi;
        PetscInt nsegment;
    };

    /** Virtual coordinates are used during the search to compute path length properties in case the simulation is not 3 dimensional */
    struct Virtualcoord {
        PetscReal x;
        PetscReal y;
        PetscReal z;
        PetscReal xdir;
        PetscReal ydir;
        PetscReal zdir;
        PetscInt current;
        PetscReal hhere;
    };

    /// Class Methods
    void RayInit();

    /** Update the coordinates of the particle using the virtual coordinates
     * Moves the particle in physical space instead of only updating the virtual coordinates
     * This function must be run on every updated particle before swarm migrate is used */
    void UpdateCoordinates(PetscInt ipart, Virtualcoord* virtualcoord, PetscReal* coord);

    /** Create a unique identifier from an array of integers.
     * This is done using the nested Cantor pairing function
     * The ray segment will always be accessed by a particle carrying an identifier so it does not need to be inverted.
     * (Unless the particles created for the solve need to find their ray segments efficiently? Maybe dont destroy the particles of the search?)
     * */
    std::string Key(Identifier id);

    eos::ThermodynamicTemperatureFunction absorptivityFunction;
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

    /// Class Constants
    const PetscReal sbc = 5.6696e-8;  //!< Stefan-Boltzman Constant (J/K)
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;
    PetscMPIInt numRanks;  //!< The number of the ranks that the simulation contains. This will be used to support global indexing.

    /// Class inputs and Variables
    PetscInt dim = 0;  //!< Number of dimensions that the domain exists within
    PetscInt nTheta;   //!< The number of angles to solve with, given by user input
    PetscInt nPhi;     //!< The number of angles to solve with, given by user input (x2)

    /**
     * Store a log used to output the required information
     */
    const std::shared_ptr<ablate::monitors::logs::Log> log;

    std::map<std::string, Segment> rays;
    std::map<PetscInt, Origin> origin;
};

}  // namespace ablate::radiation
#endif
