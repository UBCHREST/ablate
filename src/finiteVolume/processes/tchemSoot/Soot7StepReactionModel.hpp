#ifndef ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#define ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#include <iostream>
#include <cmath>
#include "utilities/constants.hpp"
#include "TChem_KineticModelData.hpp"

namespace ablate::finiteVolume::processes::tchemSoot::Soot7StepReactionModel{

    inline double solidCarbonDensity = 2000; // set to 2k kg/m^3
    inline double Ca = 3.;
    inline double Cmin = 700;
    inline double MWCarbon = 12.0107;
    inline double OxidationCollisionEfficiency = .2;
    inline double NdNuclationConversionTerm = 2./Cmin * 6.02214076e26; // this is the soot conversion term used to convert the nucleation rate for carbon to its rate for soot number density = 2 (avogadro's number)/Cmin

    inline double calculateSootDiameter(double YCarbon, double Nd){ return std::pow(6*YCarbon/ablate::utilities::Constants::pi/solidCarbonDensity/Nd,1./3.);}
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
        return 2*Ca*std::sqrt(dp)*std::sqrt(6*(1.380649e-23)*T/solidCarbonDensity)*(totalDensity*totalDensity*Nd*Nd); // the Term in the paranthesis is Boltzman's constant
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
    static void UpdateSourceWithSootMechanismRates(const Tines::value_type_1d_view<real_type, device_type>& x, const Tines::value_type_1d_view<real_type, device_type>& f, KineticModelConstData<device_type>_kmcd){
        f(YC_sInd) = 1;
        f(NddInd) = 2;
    }


    template<typename device_type>
    static void UpdateSpeciesSpecificIndices(KineticModelConstData<device_type> _kmcd) {
        auto spNames = _kmcd.speciesNames;
            YC_sInd = _kmcd.nSpec+1;
            NddInd = _kmcd.nSpec+2;
        //Find species solution vector locations for O, O2, OH, CO, H
//            std::cout << spNames(4) << std::endl;

    }

//        void UpdateSourceWithSootMechanismRates(){}

} //End TchemSoot namespace
#endif
