#ifndef ABLATELIBRARY_SOLUTIONERRORMONITOR_HPP
#define ABLATELIBRARY_SOLUTIONERRORMONITOR_HPP
#include "monitor.hpp"
#include <iostream>
#include <vector>
namespace ablate::monitors {

class SolutionErrorMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);

   public:
    enum class Scope {VECTOR, COMPONENT};
    Scope errorScope;
    enum class Norm {L2, LINF, L2_NORM};
    Norm normType;

    SolutionErrorMonitor(Scope errorScope, Norm normType);

    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorError; }

    std::vector<PetscReal> ComputeError(TS ts, PetscReal time, Vec u);
};

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream & operator << (std::ostream & os, const SolutionErrorMonitor::Scope& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream & operator >> (std::istream& is, SolutionErrorMonitor::Scope& v);

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream & operator << (std::ostream & os, const SolutionErrorMonitor::Norm& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream & operator >> (std::istream& is, SolutionErrorMonitor::Norm& v);


}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_ERRORMONITOR_HPP
