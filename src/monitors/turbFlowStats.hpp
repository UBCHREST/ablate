#ifndef ABLATELIBRARY_TURBFLOWSTATS_HPP
#define ABLATELIBRARY_TURBFLOWSTATS_HPP

#include "monitor.hpp"
#include <string>
#include "monitors/logs/log.hpp"
#include "io/interval/interval.hpp"
#include "eos/eos.hpp"
#include "domain/field.hpp"
//#include "io/serializable.hpp"

namespace ablate::monitors {

enum FieldOffset {dSum, dMult};

class TurbFlowStats : public Monitor {

   protected:
    void AddField(DM &dm, const char* nameField, const char* nameRegion, PetscInt numComp);

   private:
    const std::string fieldName;
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;
    Vec turbVec;
    DM turbDM;

    inline static const double tiny = 1e-30;
    static PetscErrorCode MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    explicit TurbFlowStats(const std::string& nameIn, const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<logs::Log> logIn = {}, std::shared_ptr<io::interval::Interval> intervalIn = {});
    PetscMonitorFunction GetPetscFunction() override { return MonitorTurbFlowStats; }
    void Register(std::shared_ptr<ablate::solver::Solver> solverIn) override;
};

}

#endif //ABLATELIBRARY_TURBFLOWSTATS_HPP