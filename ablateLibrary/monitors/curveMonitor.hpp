#ifndef ABLATELIBRARY_CURVEMONITOR_HPP
#define ABLATELIBRARY_CURVEMONITOR_HPP

#include <iostream>
#include <memory>
#include "io/interval/interval.hpp"
#include "monitor.hpp"

namespace ablate::monitors {

class CurveMonitor : public Monitor {
   private:
    const std::shared_ptr<io::interval::Interval> interval;
    const std::string filePrefix;
    inline static const std::string fileExtension = ".curve";
    /**
     * Curve output values cannot be lessthan the this minimum in order to be read into visit
     */
    inline static const PetscReal minimumOutputValue = 1E-64;

    static PetscErrorCode OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

    static void WriteToCurveFile(std::ostream& curveFile, PetscInt cStart, PetscInt cEnd, Vec cellGeomVec, const std::vector<domain::Field>& fields, DM dm, Vec vec);

   public:
    CurveMonitor(std::shared_ptr<io::interval::Interval> interval, std::string prefix);

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputCurve; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_CURVEMONITOR_HPP
