#include "raySharingRadiation.hpp"

ablate::radiation::RaySharingRadiation::RaySharingRadiation(const std::string& solverId, const std::shared_ptr<domain::Region>& region, const PetscInt raynumber,
                                                            std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, std::shared_ptr<ablate::monitors::logs::Log> log)
    : Radiation(solverId, region, raynumber, radiationModelIn, log) {}

ablate::radiation::RaySharingRadiation::~RaySharingRadiation() {}

void ablate::radiation::RaySharingRadiation::Setup(const ablate::domain::Range& cellRange, ablate::domain::SubDomain& subDomain) {
    indexLookup = ablate::domain::ReverseRange(cellRange);

    ablate::radiation::Radiation::Setup(cellRange, subDomain);
}

void ablate::radiation::RaySharingRadiation::IdentifyNewRaysOnRank(ablate::domain::SubDomain& subDomain, DM radReturn, PetscInt npoints) {
    StartEvent((GetClassType() + "::Initialize::IdentifyNewRaysOnRank").c_str());
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Declare some information associated with the field declarations */
    PetscInt* index;
    struct Identifier* identifiers;     //!< Pointer to the ray identifier information
    struct Virtualcoord* virtualcoord;  //!< Pointer to the primary (virtual) coordinate field information

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;

    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        /** Check that the particle is in a valid region */
        if (index[ipart] >= 0) {
            auto& identifier = identifiers[ipart];
            // If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory and record its index
            if (identifier.remoteRank != rank) {
                //! Get nTheta
                double theta = acos(virtualcoord[ipart].zdir);
                double phi = atan2(virtualcoord[ipart].ydir, virtualcoord[ipart].xdir);
                PetscInt ntheta = (PetscInt)(((theta / ablate::utilities::Constants::pi) * (double)nTheta) - 0.5);
                PetscInt nphi = (PetscInt)((phi / (2.0 * ablate::utilities::Constants::pi)) * (double)nPhi);

                // Update the identifier for this rank.  When it gets sent to another rank a copy will be made
                identifier.remoteRank = rank;
                // set the remoteRayId to be the next one in the way
                identifier.remoteRayId =
                    (PetscInt)indexLookup.GetAbsoluteIndex(index[ipart]) * raysPerCell + ntheta * nPhi + nphi;  //! Should be set to (absoluteCellIndex * raysPerCell + angleNumber)
                // bump the nSegment
                identifier.nSegment++;

                // Create an empty struct in the ray
                raySegments.emplace_back();

                // store this ray and information in the return data
                DMSwarmAddPoint(radReturn) >> utilities::PetscUtilities::checkError;  //!< Another solve particle is added here because the search particle has entered a new domain
                struct Identifier* returnIdentifiers;                                 //!< Pointer to the ray identifier information
                PetscInt* returnRank;                                                 //! while we are here, set the return rank.  This won't change anything until migrate is called
                DMSwarmGetField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
                    utilities::PetscUtilities::checkError;                            //!< Get the fields from the radsolve swarm so the new point can be written to them
                DMSwarmGetField(radReturn, DMSwarmField_rank, nullptr, nullptr, (void**)&returnRank) >> utilities::PetscUtilities::checkError;

                // these are only created as remote rays are identified, so we can remoteRayId for the rank
                returnIdentifiers[identifier.remoteRayId] = identifier;
                returnRank[identifier.remoteRayId] = identifier.originRank;

                DMSwarmRestoreField(radReturn, IdentifierField, nullptr, nullptr, (void**)&returnIdentifiers) >>
                    utilities::PetscUtilities::checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                DMSwarmRestoreField(radReturn, DMSwarmField_rank, nullptr, nullptr, (void**)&returnRank) >> utilities::PetscUtilities::checkError;
            }
        }
    }
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoord) >> utilities::PetscUtilities::checkError;
    EndEvent();
}

void ablate::radiation::RaySharingRadiation::ParticleStep(ablate::domain::SubDomain& subDomain, DM faceDM, const PetscScalar* faceGeomArray, DM radReturn, PetscInt npoints,
                                                PetscInt nglobalpoints) { /** Check that the particle is in a valid region */

    PetscFVFaceGeom* faceGeom;

    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Declare some information associated with the field declarations */
    PetscInt* index;
    struct Virtualcoord* virtualcoords;  //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifiers;      //!< Pointer to the ray identifier information

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    DMSwarmGetField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoords) >> utilities::PetscUtilities::checkError;
    DMSwarmGetField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;

    for (PetscInt ipart = 0; ipart < npoints; ipart++) {
        /** Check that the particle is in a valid region */
        if (index[ipart] >= 0 && subDomain.InRegion(index[ipart])) {
            StartEvent((GetClassType() + "::Initialize::GetConnectivity").c_str());
            auto& identifier = identifiers[ipart];
            // Exact the ray to reduce lookup
            auto& ray = raySegments[identifier.remoteRayId];

            /** ********************************************
             * The face stepping routine will give the precise path length of the mesh without any error. It will also allow the faces of the cells to be accounted for so that the
             * boundary conditions and the conditions at reflection can be accounted for. This will make the entire initialization much faster by only requiring a single step through each
             * cell. Additionally, the option for reflection is opened because the faces and their normals are now more easily accessed during the initialization. In the future, the carrier
             * particles may want to be given some information that the boundary label carries when the search particle happens upon it so that imperfect reflection can be implemented.
             * */

            /** Step 2: Acquire the intersection of the particle search line with the segment or face. In the case if a two dimensional mesh, the virtual coordinate in the z direction will
             * need to be solved for because the three dimensional line will not have a literal intersection with the segment of the cell. The third coordinate can be solved for in this case.
             * Here we are figuring out what distance the ray spends inside the cell that it has just registered.
             * */
            /** March over each face on this cell in order to check them for the one which intersects this ray next */
            PetscInt numberFaces;
            const PetscInt* cellFaces;
            DMPlexGetConeSize(subDomain.GetDM(), index[ipart], &numberFaces) >> utilities::PetscUtilities::checkError;
            DMPlexGetCone(subDomain.GetDM(), index[ipart], &cellFaces) >> utilities::PetscUtilities::checkError;  //!< Get the face geometry associated with the current cell
            EndEvent();

            StartEvent((GetClassType() + "::Initialize::CalculatePathLength").c_str());
            /** Check every face for intersection with the segment.
             * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
             * */
            for (PetscInt f = 0; f < numberFaces; f++) {
                PetscInt face = cellFaces[f];
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &faceGeom) >> utilities::PetscUtilities::checkError;  //!< Reads the cell location from the current cell

                /** Get the intersection of the direction vector with the cell face
                 * Use the plane equation and ray segment equation in order to get the face intersection with the shortest path length
                 * This will be the next position of the search particle
                 * */
                PetscReal path = FaceIntersect(ipart, virtualcoords, faceGeom);  //!< Use plane intersection equation by getting the centroid and normal vector of the face

                /** Step 3: Take this path if it is shorter than the previous one, getting the shortest path.
                 * The path should never be zero if the forwardIntersect check is functioning properly.
                 * */
                if (path > 0) {
                    virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? (path * 1.1) : virtualcoords[ipart].hhere;  //!< Dumb check to ensure that the path length is always updated
                    if (virtualcoords[ipart].hhere > path) {
                        virtualcoords[ipart].hhere = path;  //!> Get the shortest path length of all of the faces. The point must be in the direction that the ray is travelling in order to be valid.
                    }
                }
            }
            virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? minCellRadius : virtualcoords[ipart].hhere;
            /**
             * Only write the new path lengths if the segment belongs to this rank, otherwise it will just share.
             * Most instances the particle step won't occur on the remotely travelling particles because they are getting teleported
             * We want to make sure they don't write anything anyway
             */
            if (identifier.originRank == rank) { /** Step 1: Register the current cell index in the rays vector. The physical coordinates that have been set in the previous step / loop will be
                                                  * immediately registered. Because the ray comes from the origin, all of the cell indexes are naturally ordered from the center out
                                                  * */
                auto& raySegment = ray.emplace_back();
                raySegment.cell = index[ipart];
                raySegment.pathLength = virtualcoords[ipart].hhere;
            }
            EndEvent();
        } else {
            virtualcoords[ipart].hhere = (virtualcoords[ipart].hhere == 0) ? minCellRadius : virtualcoords[ipart].hhere;
        }
    }
    DMSwarmRestoreField(radSearch, IdentifierField, nullptr, nullptr, (void**)&identifiers) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, VirtualCoordField, nullptr, nullptr, (void**)&virtualcoords) >> utilities::PetscUtilities::checkError;
    DMSwarmRestoreField(radSearch, DMSwarmPICField_cellid, nullptr, nullptr, (void**)&index) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::radiation::RaySharingRadiation, ablate::radiation::RaySharingRadiation, "A solver for radiative heat transfer in participating media",
                 ARG(std::string, "id", "the name of the flow field"), ARG(ablate::domain::Region, "region", "the region to apply this solver."), ARG(int, "rays", "number of rays used by the solver"),
                 ARG(ablate::eos::radiationProperties::RadiationModel, "properties", "the radiation properties model"),
                 OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"));