#ifndef ABLATELIBRARY_TURBFLOWSTATS_HPP
#define ABLATELIBRARY_TURBFLOWSTATS_HPP

#include "monitor.hpp"
#include <string>
#include "monitors/logs/log.hpp"
#include "io/interval/interval.hpp"
#include "eos/eos.hpp"
#include "domain/field.hpp"
#include "io/serializable.hpp"

namespace ablate::monitors {

class TurbFlowStats : public Monitor, public io::Serializable{

    using ttf = ablate::eos::ThermodynamicFunction;

   protected:
    void AddField(DM &dm, const char* nameField, PetscInt numComp);

   private:
    const std::vector<std::string> fieldNames;
    const std::shared_ptr<ablate::eos::EOS> eos;
    const std::shared_ptr<io::interval::Interval> interval;
    const std::string name = "TurbFlowStats";
    std::vector<PetscInt> fieldTrack;
    Vec turbVec;
    DM turbDM;
    ttf densityFunc;

    //Keeps track of the categories used
    struct {
        PetscInt densitySum = 0;
        PetscInt densityDtSum = 1;
        PetscInt densityMult = 2;
        PetscInt densityDtMult = 3;
        PetscInt densitySqr = 4;
        PetscInt sum = 5;
        PetscInt sumSqr = 6;
        PetscInt favreAvg = 7;
        PetscInt rms = 8;
        PetscInt mRms = 9;

        void SetMode(PetscInt numComps) {
            densityMult = 2;
            densityDtMult = densityMult + numComps;
            densitySqr = densityDtMult + numComps;
            sum = densitySqr + numComps;
            sumSqr = sum + numComps;
            favreAvg = sumSqr + numComps;
            rms = favreAvg + numComps;
            mRms = rms + numComps;
        }
    }CatOffset, CatOrder;

    inline static const double tiny = 1e-30;
    static PetscErrorCode MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

   public:
    explicit TurbFlowStats(const std::vector<std::string> nameIn, const std::shared_ptr<ablate::eos::EOS> eosIn, std::shared_ptr<io::interval::Interval> intervalIn = {});
    PetscMonitorFunction GetPetscFunction() override { return MonitorTurbFlowStats; }
    void Register(std::shared_ptr<ablate::solver::Solver> solverIn) override;

    //Here are the serializable functions to override
    const std::string& GetId() const override {return name;}
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

};

}

#endif //ABLATELIBRARY_TURBFLOWSTATS_HPP