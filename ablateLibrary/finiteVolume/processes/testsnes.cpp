#include <petsc.h>
#include <iostream>
#include "petscsnes.h"
#include "eos/perfectGas.hpp"

int main() {
    std::shared_ptr<eos::EOS> eosGas;
    std::shared_ptr<eos::EOS> eosLiquid;
    eosGas = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}, {"Rgas", "287.0"}}));
    eosLiquid = std::make_shared<eos::PerfectGas>(std::make_shared<parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "3.2"}, {"Rgas", "100.2"}}));
    std::vector<PetscReal> conservedValues = {0.8, 3.9, 936986.7, 39.0, -78.0, 117.0},  // RHOALPHA, RHO, RHOE, RHOU, RHOV, RHOW
    std::vector<PetscReal> normal = {0.5, 0.5, 0.7071},                             // x, y, z
    PetscInt dim = 3;

    const int EULER_FIELD = 1;  // denstiyVF is [0] field
    // (densityVF, RHO, RHOE, RHOU, RHOV, RHOW)
    // decode
    PetscReal densityVF = conservedValues[0];
    PetscReal density = conservedValues[0 + EULER_FIELD];
    PetscReal totalEnergy = conservedValues[1 + EULER_FIELD] / density;

    // Get the velocity in this direction, and kinetic energy
    PetscReal normalVelocity = 0.0;
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[2 + EULER_FIELD + d] / (*density);
        normalVelocity += velocity[d] * normal[d];
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;
    internalEnergy = totalEnergy - ke;

    // Get mass fractions
    PetscReal Yg = densityVF / density;
    PetscReal Yl = (density - densityVF) / density;

    PetscReal pG;
    PetscReal pL;
    PetscReal aG;
    PetscReal aL;
    // additional equations:
    // 1/density = Yg/densityG + Yl/densityL;
    // internalEnergy = Yg*internalEnergyG + Yl*internalEnergyL;
    PetscReal eG = (*internalEnergy);
    PetscReal etG = eG + ke;
    PetscReal eL = ((*internalEnergy) - Yg * eG) / Yl;
    PetscReal etL = eL + ke;
    PetscReal rhoG = Yg * (*density);
    PetscReal rhoL = Yl / (1 / (*density) - Yg / rhoG);
    // while (PetscAbs(delp) > PerrorTol){
    // guess new eG, rhoG; calculate new eL, rhoL
    // decode the state in the eos
    eosGas->GetDecodeStateFunction()(dim, rhoG, etG, velocity, NULL, internalEnergy, aG, &pG, eosGas->GetDecodeStateContext());
    eosLiquid->GetDecodeStateFunction()(dim, rhoL, etL, velocity, NULL, internalEnergy, aL, &pL, eosLiquid->GetDecodeStateContext());
    // guess et1, rho1; return e1, a1, p1 | guess et2, rho2; return e2, a2, p2
    // compare pG, pL
    // PetscReal delp = pG - pL
    // if delp >0; choose new eG, rhoG this way
    // elseif delp <0: choose new eG, rhoG some other way
    // iterate result: densityG, densityL, internalEnergyG, internalEnergyL, p, T

    // once state defined
    PetscReal densityG = rhoG;
    PetscReal densityL = rhoL;
    PetscReal internalEnergyG = eG;
    PetscReal internalEnergyL = eL;
    PetscReal p = pG;
    PetscReal MG = normalVelocity / aG;
    PetscReal ML = normalVelocity / aL;
    PetscReal alpha = densityVF / densityG;

    std::cout << "density: " << density << "\n";
    std::count << " gas: " << densityG << "\n";
    std::cout << " liquid: " << densityL << "\n";
    //.expectedDensity = 3.9,
    //.expectedDensityG = 1.88229965,
    //.expectedDensityL = 5.391417167,
    //.expectedNormalVelocity = 33.72128712,
    //.expectedVelocity = {10.0, -20.0, 30.0},
    //.expectedInternalEnergy = 239553.0,
    //.expectedInternalEnergyG = 153788.3742,
    //.expectedInternalEnergyL = 9762.176166,
    //.expectedSoundSpeedG = 293.4646308,
    //.expectedSoundSpeedL = 262.1559091,
    //.expectedMG = 0.1149075,
    //.expectedML = 0.128630658,
    //.expectedPressure = 115790.322,
    //.expectedAlpha = 0.425012032},
}