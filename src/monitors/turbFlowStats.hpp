#ifndef ABLATELIBRARY_TURBFLOWSTATS_HPP
#define ABLATELIBRARY_TURBFLOWSTATS_HPP

#include <string>
#include "domain/field.hpp"
#include "eos/eos.hpp"
#include "fieldMonitor.hpp"
#include "io/interval/interval.hpp"

namespace ablate::monitors {

class TurbFlowStats : public FieldMonitor {
    using ttf = ablate::eos::ThermodynamicFunction;

    enum FieldPlacements { densitySum, densityDtSum, fieldsStart };
    enum SectionLabels { densityMult, densityDtMult, densitySqr, sum, sumSqr, favreAvg, rms, mRms, END };

   private:
    const std::vector<std::string> fieldNames;
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<io::interval::Interval> interval;
    ttf densityFunc;
    PetscInt step;

    static PetscErrorCode MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    explicit TurbFlowStats(const std::vector<std::string> nameIn, const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<io::interval::Interval> intervalIn = {});
    PetscMonitorFunction GetPetscFunction() override { return MonitorTurbFlowStats; }
    void Register(std::shared_ptr<ablate::solver::Solver> solverIn) override;

    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
};

}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_TURBFLOWSTATS_HPP