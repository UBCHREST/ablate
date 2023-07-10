#ifndef ABLATELIBRARY_RAYSHARINGRADIATION_H
#define ABLATELIBRARY_RAYSHARINGRADIATION_H

#include "domain/reverseRange.hpp"
#include "radiation.hpp"

namespace ablate::radiation {

class RaySharingRadiation : public ablate::radiation::Radiation {
   public:
    RaySharingRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                        std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> = {});
    ~RaySharingRadiation();

    /**
     * Creates a reverse range and initial segment collection to provide infrastructure for rays to be shared between processes.
     * @param cellRange
     * @param subDomain
     */
    void Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) override;

    /**
     * Instead of creating new ray segments every time a unique particle enters the process, alien particles will be assigned to an existing ray segment which matches their trajectory.
     * @param subDomain
     * @param radReturn
     * @param npoints
     */
    void IdentifyNewRaysOnRank(ablate::domain::SubDomain& subDomain, DM radReturn, PetscInt npoints) override;

    /**
     * This particle step will only write new cell indexes to a ray segment if the particle writing the segment is the first segment in the ray.
     * Being the first segment implies that the particle is native to the process. This way, no redundant cells are written.
     * @param subDomain
     * @param faceDM
     * @param faceGeomArray
     * @param radReturn
     * @param nlocalpoints
     * @param nglobalpoints
     */
    void ParticleStep(ablate::domain::SubDomain& subDomain, DM faceDM, const PetscScalar* faceGeomArray, DM radReturn, PetscInt nlocalpoints,
                      PetscInt nglobalpoints) override;  //!< Routine to move the particle one step

    static inline std::string GetClassType() { return "RaySharingRadiation"; }

    /**
     * The version of the boundary condition will only set the boundary condition if the particle is native to the rank.
     * This functions the same as the particle step function to prevent overwriting of the segment information by alien particles.
     * @param raySegment
     * @param index
     * @param identifier
     */
    void SetBoundary(CellSegment& raySegment, PetscInt index, Identifier identifier) override {
        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (identifier.originRank == rank) {
            raySegment.cell = index;
            raySegment.pathLength = -1;
        }
    }

   protected:
    //! used to look up from the cell id to range index
    ablate::domain::ReverseRange indexLookup;

    //! the indexes mapping to the ray id
    std::vector<PetscReal> remoteMap;
};

}  // namespace ablate::radiation

#endif  // ABLATELIBRARY_RAYSHARINGRADIATION_H
