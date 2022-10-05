#ifndef ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#define ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#include <iostream>
#include <cmath>
#include "eos/tChemSoot.hpp"
#include "utilities/constants.hpp"
#include "TChem_KineticModelData.hpp"

namespace ablate::finiteVolume::processes::tchemSoot::Soot7StepReactionModel{

    inline double Ca = 3.;
    inline double Cmin = 700;
    inline double MWCarbon = 12.0107;
    inline double OxidationCollisionEfficiency = .2;
    inline double NdNuclationConversionTerm = 2./Cmin * 6.02214076e26; // this is the soot conversion term used to convert the nucleation rate for carbon to its rate for soot number density = 2 (avogadro's number)/Cmin

    inline double calculateSootDiameter(double YCarbon, double Nd){ return std::pow(6*YCarbon/ablate::utilities::Constants::pi/ablate::eos::TChemSoot::solidCarbonDensity/Nd,1./3.);}
    inline double calculateSurfaceArea_V(double YCarbon, double Nd, double totalDensity) {
        double dp = calculateSootDiameter(YCarbon,Nd);
        return dp*dp*totalDensity*Nd;
    }

    inline double calculateNucleationReactionRate(double T, double C2H2Conc) {
        return 1000.*std::exp(-16103./T)*C2H2Conc;
    }

    inline double calculateSurfaceGrowthReactionRate(double T, double C2H2Conc, double YCarbon, double totalDensity, double Nd,double SA_V) {
        return 700.*std::exp(-10064./T)*C2H2Conc*std::sqrt(SA_V);
    }

    inline double calculateAgglomerationRate( double YCarbon, double Nd, double T,double totalDensity) {
        double dp = calculateSootDiameter(YCarbon,Nd);
        return 2*Ca*std::sqrt(dp)*std::sqrt(6*(1.380649e-23)*T/ablate::eos::TChemSoot::solidCarbonDensity)*(totalDensity*totalDensity*Nd*Nd); // the Term in the paranthesis is Boltzman's constant
    }

    inline double calculateO2OxidationRate ( double YCarbon, double Nd, double O2Conc, double totalDensity, double T,double SA_V)  {
        double ka = 200. * std::exp(-15098. / T);
        double kz = 21.3 * std::exp(2063. / T);
        double kb = 4.46e-2 * std::exp(-7650. / T);
        double kT = 1.51e6 * std::exp(-48817. / T);
        double PO2 = O2Conc * (8314.4626) * T/(101325.); // = |O2| R_u*T in (atm) the first parenthesis term is the UGC, and the second is the conversion of Pascals to atmospheres
        double xA = 1. / (1. + kT / (kb * PO2));
        return  (ka * PO2 * xA / (1 + kz * PO2) + kb * PO2 * (1 - xA) ) * SA_V;
    }

    inline double calculateOOxidationRate(double OHConc, double T,double SA_V) {
        return 0.001094 * OxidationCollisionEfficiency*std::sqrt(T)*(8314.4626)*OHConc*SA_V;
    }

    inline double calculateOHOxidationRate(double OConc, double T,double SA_V) {
        return .001044*OxidationCollisionEfficiency*std::sqrt(T)*(8314.4626)*OConc*SA_V;
    }

    inline int OInd = -1;
    inline int O2Ind = -1;
    inline int OHInd = -1;
    inline int COInd = -1;
    inline int HInd = -1;
    inline int YC_sInd = -1;
    inline int NddInd = -1;
    template<typename device_type>
    KOKKOS_INLINE_FUNCTION
    static void UpdateSourceWithSootMechanismRates(const Tines::value_type_1d_view<real_type, device_type>& x, const Tines::value_type_1d_view<real_type, device_type>& f, const real_type densityTotal, KineticModelConstData<device_type>_kmcd){
        real_type SVF = x(YC_sInd)*densityTotal/eos::TChemSoot::solidCarbonDensity;
        real_type O_SVF = 1-SVF;
        //Scale all Gas Source terms initially by their represented volume fraction
        for(int i =0; i < _kmcd.nSpec; i++) {
            f(i+1) *= O_SVF;
        }
        //Zero Out the temperature source since we technically aren't solving it
//        f(0) *= O_SVF;
        f(0) = 0;
        //The temperature equations is a behemoth, need Cv's of the gas, internal energies of each gas species, density and spec gas constant of the gas
        double t = x(0);
        double Yc = x(YC_sInd);
        double gasDensity = (1-Yc)/(1/densityTotal-Yc/ablate::eos::TChemSoot::solidCarbonDensity);
        double LHS = (1-Yc)*CV_GAS + Yc*(ablate::eos::TChemSoot::CarbonEnthalpy_R_T(t)*_kmcd.Runiv*t/ablate::eos::TChemSoot::MWCarbon - gasDensity/ablate::eos::TChemSoot::solidCarbonDensity*gasGasConstant);
        double RHS = f(0)*CV_Gas*(1-Yc); //First term reweights the gas terms igniting
        //Add in the Soot Reaction Sources
        f(YC_sInd) = 0;
        f(NddInd) = 1000;
    }


    template<typename device_type>
    static void UpdateSpeciesSpecificIndices(std::vector<std::string> species) {
        YC_sInd = species.size();
        NddInd = species.size()+1;
        //Find species solution vector locations for O, O2, OH, CO, H
        //Brute forcing this in here for now
        for(int sp = 1; sp < (int)species.size();sp++)

            if(species[sp] == "O")
                OInd = sp;
            else if (species[sp] == "O2")
                O2Ind = sp;
            else if (species[sp] == "OH")
                OHInd = sp;
            else if (species[sp] == "CO")
                COInd = sp;
            else if (species[sp] == "H")
                HInd = sp;

    }

//        void UpdateSourceWithSootMechanismRates(){}

} //End TchemSoot namespace
#endif
