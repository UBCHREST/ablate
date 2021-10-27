#ifndef ABLATELIBRARY_SOLUTIONERRORMONITOR_HPP
#define ABLATELIBRARY_SOLUTIONERRORMONITOR_HPP
#include <iostream>
#include <monitors/logs/log.hpp>
#include <vector>
#include "monitor.hpp"

namespace ablate::monitors {

class SolutionErrorMonitor : public Monitor {
   public:
    enum class Scope { VECTOR, COMPONENT };
    enum class Norm { L2, LINF, L2_NORM };

   private:
    static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);
    Scope errorScope;
    Norm normType;
    const std::shared_ptr<logs::Log> log;

   public:
    SolutionErrorMonitor(Scope errorScope, Norm normType, std::shared_ptr<logs::Log> log = {});

    PetscMonitorFunction GetPetscFunction() override { return MonitorError; }

    std::vector<PetscReal> ComputeError(TS ts, PetscReal time, Vec u);
};

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream& operator<<(std::ostream& os, const SolutionErrorMonitor::Scope& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, SolutionErrorMonitor::Scope& v);

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream& operator<<(std::ostream& os, const SolutionErrorMonitor::Norm& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, SolutionErrorMonitor::Norm& v);

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_ERRORMONITOR_HPP
