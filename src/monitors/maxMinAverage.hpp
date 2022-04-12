#ifndef ABLATELIBRARY_MAXMINAVERAGE_HPP
#define ABLATELIBRARY_MAXMINAVERAGE_HPP

#include <memory>
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {

class MaxMinAverage : public Monitor {
   private:
    static PetscErrorCode MonitorMaxMinAverage(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);
    const std::string fieldName;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;

   public:
    explicit MaxMinAverage(const std::string& fieldName, std::shared_ptr<logs::Log> log = {}, std::shared_ptr<io::interval::Interval> interval = {});

    PetscMonitorFunction GetPetscFunction() override { return MonitorMaxMinAverage; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_MAXMINAVERAGE_HPP
