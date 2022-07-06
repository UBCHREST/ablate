#ifndef ABLATELIBRARY_TURBFLOWSTATS_HPP
#define ABLATELIBRARY_TURBFLOWSTATS_HPP

#include "monitor.hpp"
#include <string>
#include "monitors/logs/log.hpp"
#include "io/interval/interval.hpp"


namespace ablate::monitors {

class TurbFlowStats : public Monitor {

   private:
    const std::string fieldName;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;
    static PetscErrorCode MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    explicit TurbFlowStats(const std::string nameIn, std::shared_ptr<logs::Log> logIn = {}, std::shared_ptr<io::interval::Interval> intervalIn = {});
    PetscMonitorFunction GetPetscFunction() override { return MonitorTurbFlowStats; }
};

}

#endif //ABLATELIBRARY_TURBFLOWSTATS_HPP