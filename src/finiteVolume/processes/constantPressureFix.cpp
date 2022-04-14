#include "constantPressureFix.hpp"

ablate::finiteVolume::processes::ConstantPressureFix::ConstantPressureFix(std::shared_ptr<eos::EOS> eos, double pressure) : pressure(pressure), eos(eos) {}

void ablate::finiteVolume::processes::ConstantPressureFix::Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) {
    fv.RegisterPostEvaluate([](TS ts, ablate::solver::Solver&){

    });
}
