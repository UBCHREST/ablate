#include "virtualTcp.hpp"

ablate::monitors::VirtualTcp::~VirtualTcp() {

}

void ablate::monitors::VirtualTcp::Register(std::shared_ptr<solver::Solver> solver) { Monitor::Register(solver); }

// TODO: Get the ray tracing models (possibly a vector of models) and initialize them. Use the solve to compute the intensities with different radiation properties models.

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::monitors::Monitor, ablate::monitors::VirtualTcp, "Outputs TCP information to the serializer.", ARG(ablate::radiation::Radiation, "radiation", "ray tracing solver to write information to the boundary faces."));
