#include "raySharingRadiation.hpp"

void ablate::radiation::RaySharingRadiation::ParticleStep(ablate::domain::SubDomain& subDomain, PetscSF cellSF, DM faceDM, const PetscScalar* faceGeomArray, PetscInt stepcount) {
    PetscInt npoints = 0;
    PetscInt nglobalpoints = 0;
    PetscInt nsolvepoints = 0;  //!< Counts the solve points in the current domain. This will be adjusted over the course of the loop.
    PetscInt ipart = -1;

    DMSwarmGetLocalSize(radsearch, &npoints) >> checkError;
    DMSwarmGetSize(radsearch, &nglobalpoints) >> checkError;

    PetscFVFaceGeom* faceGeom;

    PetscInt index;
    PetscMPIInt rank = 0;
    MPI_Comm_rank(subDomain.GetComm(), &rank);

    /** Declare some information associated with the field declarations */
    struct Virtualcoord* virtualcoord;   //!< Pointer to the primary (virtual) coordinate field information
    struct Identifier* identifier;       //!< Pointer to the ray identifier information
    struct Carrier* carrier;             //!< Pointer to the ray carrier information
    struct Identifier* solveidentifier;  //!< Pointer to the ray identifier information
    struct Identifier* access;           //!< Pointer to the ray identifier information

    /** Get all of the ray information from the particle
     * Get the ntheta and nphi from the particle that is currently being looked at. This will be used to identify its ray and calculate its direction. */
    DMSwarmGetField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmGetField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;

    PetscInt nFound;
    const PetscInt* point = nullptr;
    const PetscSFNode* cell = nullptr;
    PetscSFGetGraph(cellSF, nullptr, &nFound, &point, &cell) >> checkError;  //!< Using this to get the petsc int cell number from the struct (SF)

    for (PetscInt ip = 0; ip < npoints; ip++) {
        ipart++;                                                                         //!< USE IP TO DEAL WITH DMLOCATE POINTS, USE IPART TO DEAL WITH PARTICLES
        if (nFound > -1 && cell[ip].index >= 0 && subDomain.InRegion(cell[ip].index)) {  // TODO: How the carrier particles are created will depend on whether there is ray sharing or not. Make a
                                                                                         // method for the creation of carrier particles. Add an override function in the ray sharing implementation.
            index = cell[ip].index;  //!< If this is a surface implementation, then the search particle should never actually enter the boundary cell

            /** If this local rank has never seen this search particle before, then it needs to add a new ray segment to local memory
             * Hash the identifier into a key value that can be used in the map
             * We should only iterate the identifier of the search particle (/ add a solver particle) if the point is valid in the domain and is being used
             * */
            if (presence.count(Key(&identifier[ipart])) == 0) {  //!< IF THIS RAYS VECTOR IS EMPTY FOR THIS DOMAIN, THEN THE PARTICLE HAS NEVER BEEN HERE BEFORE. THEREFORE, ITERATE THE NDOMAINS BY 1.
                identifier[ipart].nsegment++;                    //!< The particle has passed through another domain!
                presence[Key(&identifier[ipart])] = true;
                DMSwarmAddPoint(radsolve) >> checkError;  //!< Another solve particle is added here because the search particle has entered a new domain

                DMSwarmGetLocalSize(radsolve, &nsolvepoints) >> checkError;  //!< Recalculate the number of solve particles so that the last one in the list can be accessed.

                DMSwarmGetField(radsolve, "identifier", nullptr, nullptr, (void**)&solveidentifier) >> checkError;  //!< Get the fields from the radsolve swarm so the new point can be written to them
                DMSwarmGetField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier) >> checkError;
                DMSwarmGetField(radsolve, "access", nullptr, nullptr, (void**)&access) >> checkError;

                PetscInt newpoint = nsolvepoints - 1;           //!< This must be replaced with the index of whatever particle there is. Maybe the last index?
                solveidentifier[newpoint] = identifier[ipart];  //!< Give the particle an identifier which matches the particle it was created with
                /** Create a new 'access identifier' and set it equal to the identifier of the current cell which the search particle is occupying */
                access[newpoint].origin = rank;                      //!< The origin should be the current rank
                access[newpoint].iCell = index;                      //!< The index that the particle is currently occupying
                access[newpoint].ntheta = identifier[ipart].ntheta;  //!< The angle of the ray we want
                access[newpoint].nphi = identifier[ipart].nphi;      //!< The angle of the ray we want
                access[newpoint].nsegment = 1;                       //!< The access identifier should always point to a native rank (segment == 1)
                // The cell index and rank of the access identifier will be equal to the current, but the ray angle number will be equal to that of the identifier. Segment number == 1
                // All inputs of information will come from the ray segment of the access identifier
                carrier[newpoint].Krad = 1;  //!< The new particle gets an empty carrier because it is holding no information yet (Krad must be initialized to 1 here: everything is init 0)

                /** Send the search particle to the end of the ray that it has picked up. This will speed up the process of stepping the particle out of the domain as much as possible.
                 * When the search particle finds a ray segment to draw from in the new domain, it will need to travel to the location of the last cell in this segment.
                 * The segment will likely not be complete yet, so the search particle will need to step without picking up cells until it reaches the end of this partition and reaches a new
                 * partition The particle should exit the domain in this fashion, leaving a carrier particle with access to the other local ray segment, and no unique segment to be identified to
                 * it. The carrier particle will have an identifier to control its travel, and an identifier to control where it draws information from. These identifiers in effect point to the
                 * same ray.
                 * If the particle is native to this domain, then we don't want to mess with it at all because it's at the beginning of building a ray.
                 */
                if (access[newpoint].origin != identifier[ipart].origin) {
                    PetscReal centroid[3];
                    PetscInt numPoints = static_cast<int>(rays[Key(&access[newpoint])].cells.size());
                    DMPlexComputeCellGeometryFVM(subDomain.GetDM(), rays[Key(&access[newpoint])].cells[numPoints - 1], nullptr, centroid, nullptr) >>
                        checkError;                                                                        //!< Get the cell center of the last cell in the ray segment
                    virtualcoord[ipart].x = centroid[0] + (virtualcoord[ipart].xdir * 2 * minCellRadius);  //!< Offset from the centroid slightly so they sit in a cell if they are on its face.
                    virtualcoord[ipart].y = centroid[1] + (virtualcoord[ipart].ydir * 2 * minCellRadius);
                    virtualcoord[ipart].z = centroid[2] + (virtualcoord[ipart].zdir * 2 * minCellRadius);
                }

                DMSwarmRestoreField(radsolve, "identifier", nullptr, nullptr, (void**)&solveidentifier) >> checkError;  //!< The fields must be returned so that the swarm can be updated correctly?
                DMSwarmRestoreField(radsolve, "carrier", nullptr, nullptr, (void**)&carrier) >> checkError;
                DMSwarmRestoreField(radsolve, "access", nullptr, nullptr, (void**)&access) >> checkError;
            }

            /** ********************************************
             * The face stepping routine will give the precise path length of the mesh without any error. It will also allow the faces of the cells to be accounted for so that the
             * boundary conditions and the conditions at reflection can be accounted for. This will make the entire initialization much faster by only requiring a single step through each
             * cell. Additionally, the option for reflection is opened because the faces and their normals are now more easily accessed during the initialization. In the future, the carrier
             * particles may want to be given some information that the boundary label carries when the search particle happens upon it so that imperfect reflection can be implemented.
             * */

            /** Step 1: Register the current cell index in the rays vector. The physical coordinates that have been set in the previous step / loop will be immediately registered.
             * */
            if (identifier[ipart].nsegment == 1) rays[Key(&identifier[ipart])].cells.push_back(index);

            /** Step 2: Acquire the intersection of the particle search line with the segment or face. In the case if a two dimensional mesh, the virtual coordinate in the z direction will
             * need to be solved for because the three dimensional line will not have a literal intersection with the segment of the cell. The third coordinate can be solved for in this case.
             * Here we are figuring out what distance the ray spends inside the cell that it has just registered.
             * */
            /** March over each face on this cell in order to check them for the one which intersects this ray next */
            PetscInt numberFaces;
            const PetscInt* cellFaces;
            DMPlexGetConeSize(subDomain.GetDM(), index, &numberFaces) >> checkError;
            DMPlexGetCone(subDomain.GetDM(), index, &cellFaces) >> checkError;  //!< Get the face geometry associated with the current cell
            PetscReal path;

            /** Check every face for intersection with the segment.
             * The segment with the shortest path length for intersection will be the one that physically intercepts with the cell face and not with the nonphysical plane beyond the face.
             * */
            for (PetscInt f = 0; f < numberFaces; f++) {
                PetscInt face = cellFaces[f];
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &faceGeom) >> checkError;  //!< Reads the cell location from the current cell

                /** Get the intersection of the direction vector with the cell face
                 * Use the plane equation and ray segment equation in order to get the face intersection with the shortest path length
                 * This will be the next position of the search particle
                 * */
                path = FaceIntersect(ipart, virtualcoord, faceGeom);  //!< Use plane intersection equation by getting the centroid and normal vector of the face

                /** Step 3: Take this path if it is shorter than the previous one, getting the shortest path.
                 * The path should never be zero if the forwardIntersect check is functioning properly.
                 * */
                if (path > 0) {
                    virtualcoord[ipart].hhere = (virtualcoord[ipart].hhere == 0) ? (path * 1.1) : virtualcoord[ipart].hhere;  //!< Dumb check to ensure that the path length is always updated
                    if (virtualcoord[ipart].hhere > path) {
                        virtualcoord[ipart].hhere = path;  //!> Get the shortest path length of all of the faces. The point must be in the direction that the ray is travelling in order to be valid.
                    }
                }
            }
            virtualcoord[ipart].hhere = (virtualcoord[ipart].hhere == 0) ? minCellRadius : virtualcoord[ipart].hhere;
            if (identifier[ipart].nsegment == 1) rays[Key(&identifier[ipart])].h.push_back(virtualcoord[ipart].hhere);  //!< Add this space step if the current index is being added.
        } else {
            virtualcoord[ipart].hhere = (virtualcoord[ipart].hhere == 0) ? minCellRadius : virtualcoord[ipart].hhere;
        }
    }
    DMSwarmRestoreField(radsearch, "identifier", nullptr, nullptr, (void**)&identifier) >> checkError;
    DMSwarmRestoreField(radsearch, "virtual coord", nullptr, nullptr, (void**)&virtualcoord) >> checkError;
}