#ifndef ABLATELIBRARY_CURVEMONITOR_HPP
#define ABLATELIBRARY_CURVEMONITOR_HPP

#include <petsc.h>
#include "finiteVolume/finiteVolume.hpp"
#include <vector>
#include "monitor.hpp"
namespace ablate::monitors {

class CurveMonitor : public Monitor {
   private:
    // init variables
    const PetscInt interval;
    const std::vector<double> start;
    const std::vector<double> end;
    const std::vector<std::string> outputFields;
    const std::vector<std::string> outputAuxFields;

    const std::string filePrefix;
    inline static const std::string fileExtension = ".curve";

    // working variables
    PetscInt outputIndex = 0; /*keep track of the local cell number we are outputting*/
    std::vector<PetscInt> indexLocations;
    std::vector<PetscReal> distanceAlongLine;
    std::shared_ptr<ablate::finiteVolume::FiniteVolume> flow;

    static PetscErrorCode OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

   public:
    CurveMonitor(int interval, std::string prefix, std::vector<double> start, std::vector<double> end, std::vector<std::string> outputFields, const std::vector<std::string> outputAuxFields);

    void Register(std::shared_ptr<Monitorable>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputCurve; }
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_CURVEMONITOR_HPP