#include "surface.hpp"
#include <stdexcept>
#include "utilities/petscError.hpp"

ablate::mathFunctions::geom::Surface::Surface(std::filesystem::path meshPath, std::vector<double> insideValues, std::vector<double> outsideValues, int egadsVerboseLevel)
    : Geometry(insideValues, outsideValues) {
    // Create a surface from the meshFile
    if (!exists(meshPath)) {
        throw std::runtime_error("Cannot locate ablate::mathFunctions::geom::Surface::Surface file " + meshPath.string());
    }
    EG_open(&context) >> checkError;
    EG_setOutLevel(context, egadsVerboseLevel);
    EG_loadModel(context, 0, meshPath.c_str(), &model) >> checkError;
}
ablate::mathFunctions::geom::Surface::~Surface() {
    if (model) {
        EG_deleteObject(model);
    }
    if (context) {
        EG_close((ego)context);
    }
}
bool ablate::mathFunctions::geom::Surface::InsideGeometry(const double *xyz, const int &ndims, const double &time) const {
    // Get all the bodies in this domain
    ego geom, *bodies;
    int numberBodies;
    int oclass, mtype, *senses;

    EG_getTopology(model, &geom, &oclass, &mtype, nullptr, &numberBodies, &bodies, &senses) >> checkError;
    PetscPrintf(PETSC_COMM_SELF, " Number of BODIES (nbodies): %d \n", numberBodies) >> checkError;

    // Make sure always supply 3D array
    double coord[3] = {0.0, 0.0, 0.0};
    PetscArraycpy(coord, xyz, ndims);

    // March over each body
    bool inside = false;
    for (int b = 0; b < numberBodies; b++) {
        int result = EG_inTopology(bodies[b], coord);
        if (result == 0) {
            inside = true;
        } else if (result < 0) {
            throw std::runtime_error("EGADS Error Code  " + std::to_string(result) + " reported.");
        }
    }

    return inside;
    return false;
}
