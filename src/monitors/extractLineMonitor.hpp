#ifndef ABLATELIBRARY_EXTRACTLINEMONITOR_HPP
#define ABLATELIBRARY_EXTRACTLINEMONITOR_HPP

#include <petsc.h>
#include <vector>
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "monitor.hpp"
namespace ablate::monitors {

class ExtractLineMonitor : public Monitor {
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
    std::shared_ptr<ablate::finiteVolume::FiniteVolumeSolver> flow;

    static PetscErrorCode OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

   public:
    ExtractLineMonitor(int interval, std::string prefix, std::vector<double> start, std::vector<double> end, std::vector<std::string> outputFields, const std::vector<std::string> outputAuxFields);

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputCurve; }
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_EXTRACTLINEMONITOR_HPP