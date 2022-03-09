#ifndef ABLATELIBRARY_PROBE_HPP
#define ABLATELIBRARY_PROBE_HPP

#include <petsc.h>
#include <vector>

namespace ablate::monitors::probes {

/**
 * Helper struct to represent each location in a probeInitializer
 */
struct Probe {
    std::string name;
    std::vector<PetscReal> location;

    Probe(std::string name, std::vector<PetscReal> location) : name(std::move(name)), location(std::move(location)) {}
};
}  // namespace ablate::monitors::probes

#endif  // ABLATELIBRARY_PROBE_HPP
