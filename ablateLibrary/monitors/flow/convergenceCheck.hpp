#ifndef ABLATELIBRARY_CONVERGENCECHECK_HPP
#define ABLATELIBRARY_CONVERGENCECHECK_HPP
#include "flowMonitor.hpp"

namespace ablate::monitors::flow {
class ConvergenceCheck : public FlowMonitor {
   public:
    virtual ~ConvergenceCheck() = default;
    explicit ConvergenceCheck(int interval);

    PetscMonitorFunction GetPetscFunction() override { return CheckConvergence; }

   private:
    const int interval;  // dictates how often the check is run where 0 is only the first and last step
    std::shared_ptr<ablate::flow::Flow> flow;

    void Register(std::shared_ptr<ablate::flow::Flow>) override;

    static PetscErrorCode CheckConvergence(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);
};
}  // namespace ablate::monitors::flow

#endif  // ABLATELIBRARY_CONVERGENCECHECK_HPP
