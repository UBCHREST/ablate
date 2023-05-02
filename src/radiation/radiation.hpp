#ifndef ABLATELIBRARY_RADIATION_HPP
#define ABLATELIBRARY_RADIATION_HPP

#include <petsc/private/dmimpl.h>
#include <petscdm.h>
#include <petscdmswarm.h>
#include <petscsf.h>
#include <memory>
#include <set>
#include <utility>
#include "eos/radiationProperties/radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "io/interval/interval.hpp"
#include "monitors/logs/log.hpp"
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"
#include "utilities/constants.hpp"
#include "utilities/loggable.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

namespace ablate::radiation {

class Radiation : protected utilities::Loggable<Radiation> {  //!< Cell solver provides cell based functionality, right hand side function compatibility with
                                                              //!< finite element/ volume, loggable allows for the timing and tracking of events

   public:
    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param rayNumber
     * @param options other options
     */
    Radiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber, std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn,
              std::shared_ptr<ablate::monitors::logs::Log> = {});

    virtual ~Radiation();

    /** Identifiers are carrying by both the search and solve particles in order to associate them with their origins and ray segments
     * In the search particles, nsegment iterates based on how many domains the search particle has crossed.
     * In the solve particle, nsegment remains constant as it ties the particle to its specific order in the ray */
    struct Identifier {
        //! the rank for the start of the ray
        PetscInt originRank;
        //! The local ray id 'index'
        PetscInt originRayId;
        //! the remote rank (may be same as originating) for this segment id
        PetscInt remoteRank;
        //! The local ray id 'index'
        PetscInt remoteRayId;
        //! The number of segments away from the origin, zero on the origin
        PetscInt nSegment;
    };

    /** Carriers are attached to the solve particles and bring ray information from the local segments to the origin cells
     * They are transported directly from the segment to the origin. They carry only the values that the Segment computes and not the spatial information necessary to  */
    struct Carrier {
        PetscReal Ij = 0;    //!< Black body source for the segment. Make sure that this is reset every solve after the value has been transported.
        PetscReal Krad = 1;  //!< Absorption for the segment. Make sure that this is reset every solve after the value has been transported.
    };

    /**
     * Returns the black body intensity for a given temperature and emissivity
     * @param temperature
     * @param refractiveIndex
     * @return
     */
    static inline PetscReal GetBlackBodyTotalIntensity(PetscReal temperature, const PetscReal refractiveIndex) {
        return refractiveIndex * refractiveIndex * ablate::utilities::Constants::sbc * temperature * temperature * temperature * temperature / ablate::utilities::Constants::pi;
    }

    /**
     * Returns the black body intensity at a specific wavelength. This must be integrated over a bandwidth.
     *
     * @param temperature
     * @param wavelength
     * @param refractiveIndex
     * @return
     */
    static inline PetscReal GetBlackBodyWavelengthIntensity(const PetscReal temperature, const PetscReal wavelength, const PetscReal refractiveIndex) {
        PetscReal E_lambda = (2 * ablate::utilities::Constants::pi * ablate::utilities::Constants::h * ablate::utilities::Constants::c * ablate::utilities::Constants::c);
        E_lambda /= refractiveIndex * refractiveIndex * wavelength * wavelength * wavelength * wavelength * wavelength;
        E_lambda /= exp((ablate::utilities::Constants::h * ablate::utilities::Constants::c) / (refractiveIndex * wavelength * ablate::utilities::Constants::k * temperature)) - 1;
        E_lambda /= ablate::utilities::Constants::pi;

        return E_lambda;
    }

    /**
     * Represents the name of the class for logging and other utilities
     * @return
     */
    static inline std::string GetClassType() { return "Radiation"; }

    /** SubDomain Register and Setup **/
    virtual void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain);

    /**
     * @param cellRange The range of cells for which rays are initialized
     */
    virtual void Initialize(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain);

    /**
     * Compute total intensity (pre computed gains + current loss) with
     * @param index the current index in the cell range (note, this is not the cell/face id)
     * @param cellRange the cell/face range used in setup/initialize
     * @param temperature the temperature of the cell or face
     * @param kappa the absorptivity of the cell
     * @return
     */
    inline void GetIntensity(PetscReal* intensity, PetscInt index, const domain::Range& cellRange, PetscReal temperature, PetscReal kappa) {
        for (int i = 0; i < (int)absorptivityFunction.propertySize; ++i) {                                                  // Compute the losses
            PetscReal netIntensity = -4.0 * ablate::utilities::Constants::pi * GetBlackBodyTotalIntensity(temperature, 1);  //! This should take the emission as an input (if non-grey)

            // add in precomputed gains
            netIntensity += evaluatedGains[absorptivityFunction.propertySize * (index - cellRange.start) + i];

            // scale by kappa
            netIntensity *= kappa;

            intensity[i] = abs(netIntensity) > ablate::utilities::Constants::large ? ablate::utilities::Constants::large * PetscSignReal(netIntensity) : netIntensity;
        }
    }

    //  Add a class called "radiation surface properties" which is used to compute the amount of intensity absorbed by the material. This should be an optional input in the radiation base class.
    // GetIntensity should compute this, because otherwise the losses will not be accounted for.
    // Just embed a function within the GetIntensity function that "Computes surface properties" and decides how much of the radiation to do.
    // Also. the losses are wavelength dependent based on what portion of the black body temperature they are emitting.
    // Then, the camera properties will be able to be viewed as a radiation surface property.
    // The radiation solver can take a vector of radiation surface properties.
    // Each radiation surface property will be evaluated on the EvaluateGains call.
    // Each property can be output separately.

    inline std::string GetId() { return solverId; };

    inline eos::ThermodynamicTemperatureFunction GetAbsorptionFunction() { return absorptivityFunction; };

    /** Evaluates the ray intensity from the domain to update the effects of irradiation. Does not impact the solution unless the solve function is called again.
     * */
    void EvaluateGains(Vec solVec, ablate::domain::Field temperatureField, Vec auxVec);

    /** Determines the next location of the search particles during the initialization
     * */
    virtual void ParticleStep(ablate::domain::SubDomain& subDomain, DM faceDM, const PetscScalar* faceGeomArray, DM radReturn, PetscInt nlocalpoints,
                              PetscInt nglobalpoints);  //!< Routine to move the particle one step

    //! If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory and record its index
    void IdentifyNewRaysOnRank(domain::SubDomain& subDomain, DM radReturn, PetscInt nlocalpoints);

    /** Determines what component of the incoming radiation should be accounted for when evaluating the irradiation for each ray.
     * Dummy function that doesn't do anything unless it is overridden by the surface implementation
     * */
    virtual PetscReal SurfaceComponent(const PetscReal normal[], PetscInt iCell, PetscInt nphi, PetscInt ntheta);

    //! provide access to the model used to provided the absorptivity function
    inline std::shared_ptr<eos::radiationProperties::RadiationModel> GetRadiationModel() { return radiationModel; }

   protected:
    //! DM which the search particles occupy.  This representations the physical particle in space
    DM radSearch = nullptr;

    //! Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec faceGeomVec = nullptr;
    Vec cellGeomVec = nullptr;

    //! create a data type to simplify moving the carrier
    MPI_Datatype carrierMpiType;

    /** CellSegment belong to the local maps and hold all of the local information about the ray segments both during the search and the solve */
    struct CellSegment {
        //!< Stores the cell indices of the segment locally.
        PetscInt cell;
        //!< Stores the space steps of the segment locally.
        PetscReal pathLength;
    };

    /** Virtual coordinates are used during the search to compute path length properties in case the simulation is not 3 dimensional */
    struct Virtualcoord {
        PetscReal x;
        PetscReal y;
        PetscReal z;
        PetscReal xdir;
        PetscReal ydir;
        PetscReal zdir;
        PetscReal hhere;
    };

    /// Class Methods
    /** Returns the forward path length of a travelling particle with any face.
     * The function will return zero if the intersection is not in the direction of travel.
     *x
     * @param virtualcoord the struct containing particle position information
     * @param face the struct containing information about a cell face
     */
    PetscReal FaceIntersect(PetscInt ip, Virtualcoord* virtualcoord, PetscFVFaceGeom* face) const;  //!< Returns the distance away from a virtual coordinate at which its path intersects a line.

    /** Update the coordinates of the particle using the virtual coordinates
     * Moves the particle in physical space instead of only updating the virtual coordinates
     * This function must be run on every updated particle before swarm migrate is used
     * @param ipart the particle index which is being updated
     * @param virtualcoord the virtual coordinate field which is being read from
     * @param coord the DMSwarm coordinate field which is being written to
     * @param adv a multiple of the minimum cell radius by which to advance the DMSwarm coordinates ahead of the virtual coordinates
     * */
    void UpdateCoordinates(PetscInt ipart, Virtualcoord* virtualcoord, PetscReal* coord, PetscReal adv) const;

    /**
     * Delete the particles that are outside of the domain
     * This is used in the initialization for ray tracing solvers which rely on domain bounds to inform the placement of particles around the boundary regions.
     * @param subDomain
     */
    void DeleteOutOfBounds(ablate::domain::SubDomain& subDomain);

    /// Class inputs and Variables
    PetscInt dim = 0;  //!< Number of dimensions that the domain exists within
    PetscInt nTheta;   //!< The number of angles to solve with, given by user input
    PetscInt nPhi;     //!< The number of angles to solve with, given by user input (x2)
    PetscReal minCellRadius{};

    //! store the local rays identified on this rank.  This includes rays that do and do not originate on this rank
    std::vector<std::vector<CellSegment>> raySegments;

    //! the calculation over each of the remoteRays. indexed over remote ray
    std::vector<Carrier> raySegmentsCalculations;

    //! store the number of originating rays
    PetscInt numberOriginRays;

    //! store the number of originating rays cells
    PetscInt numberOriginCells;

    //! the number of rays per cell
    PetscInt raysPerCell;

    //! store the number of ray segments for each originating on this rank.  This may be zero
    std::vector<unsigned short int> raySegmentsPerOriginRay;

    //! a vector of raySegment information for every local/remote ray segment ordered as ray, segment
    std::vector<Carrier> raySegmentSummary;

    //! the factor for each origin ray
    std::vector<PetscReal> gainsFactor;

    //! size up the evaluated gains, this index is based upon order of the requested cells
    std::vector<PetscScalar> evaluatedGains;

    //! Store the petscSF that is used for pulling remote ray calculation
    PetscSF remoteAccess;

    //! the name of this solver
    std::string solverId;

    //! the region for which this solver applies
    const std::shared_ptr<domain::Region> region;

    //! model used to provided the absorptivity function
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

    //! hold a pointer to the absorptivity function
    eos::ThermodynamicTemperatureFunction absorptivityFunction;
    eos::ThermodynamicTemperatureFunction emissivityFunction;

    // !Store a log used to output the required information
    const std::shared_ptr<ablate::monitors::logs::Log> log = nullptr;
    static inline constexpr char IdentifierField[] = "identifier";
    static inline constexpr char VirtualCoordField[] = "virtual coord";
};
/**
 * provide write for the id
 * @param os
 * @param id
 * @return
 */
std::ostream& operator<<(std::ostream& os, const Radiation::Identifier& id);

}  // namespace ablate::radiation
#endif
