#include "probe.hpp"

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::monitors::probes::Probe, ablate::monitors::probes::Probe, "Probe specification struct", ARG(std::string, "name", "name of the probe"),
                 ARG(std::vector<double>, "location", "the probe location"));