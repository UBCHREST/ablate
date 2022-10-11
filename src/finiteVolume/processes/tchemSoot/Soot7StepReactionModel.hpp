#ifndef ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#define ABLATELIBRARY_SOOT7STEPREACTIONMODEL_HPP
#include <iostream>
#include <cmath>
#include "eos/tChemSoot.hpp"
#include "utilities/constants.hpp"
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "TChem_SpecificHeatCapacityConsVolumePerMass.hpp"
#include "TChem_Impl_MolarWeights.hpp"

namespace ablate::finiteVolume::processes::tchemSoot::Soot7StepReactionModel{

    inline double Ca = 3.;
    inline double Cmin = 700;
    inline double OxidationCollisionEfficiency = .2;
    inline double NdNuclationConversionTerm = 2./Cmin * 6.02214076e26; // this is the soot conversion term used to convert the nucleation rate for carbon to its rate for soot number density = 2 (avogadro's number)/Cmin

    inline double calculateSootDiameter(double YCarbon, double Nd){
        return std::pow(6*YCarbon/ablate::utilities::Constants::pi/ablate::eos::TChemSoot::solidCarbonDensity/(Nd+ablate::utilities::Constants::tiny),1./3.);
    }
    inline double calculateSurfaceArea_V(double YCarbon, double Nd, double totalDensity) {
        double dp = calculateSootDiameter(YCarbon,Nd);
        return ablate::utilities::Constants::pi*dp*dp*totalDensity*Nd;
    }

    inline double calculateNucleationReactionRate(double T, double C2H2Conc,double fv) {
        return 1000.*std::exp(-16103./T)*C2H2Conc * (1-fv);
    }

    inline double calculateSurfaceGrowthReactionRate(double T, double C2H2Conc, double SA_V) {
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

    inline double calculateOOxidationRate(double OConc, double T,double SA_V,double fv) {
        return 0.001094 * OxidationCollisionEfficiency*std::sqrt(T)*(8314.4626)*OConc * SA_V*(1-fv);
    }

    inline double calculateOHOxidationRate(double OHConc, double T,double SA_V,double fv) {
        return .001044*OxidationCollisionEfficiency*std::sqrt(T)*(8314.4626)*OHConc * SA_V*(1-fv);
    }

    inline int OInd = -1;
    inline int O2Ind = -1;
    inline int OHInd = -1;
    inline int COInd = -1;
    inline int HInd = -1;
    inline int H2Ind =-1;
    inline int C2H2Ind = -1;
    inline int YC_sInd = -1;
    inline int NddInd = -1;

    template<typename device_type>
    static void UpdateSpeciesSpecificIndices(std::vector<std::string> species) {
        //There is some common Inferences made here in relating the species index in the species string to the index that we expect the species to be at in the dependent variable in the ode, i.e. x.
        //Temperature equation is index 0, then there are N_species-1 gas phase species, and then the Solid Carbon Equation.
        //So the Soot index is the at the N_species location and the NDD index is 1 more
        YC_sInd = species.size();
        NddInd = species.size()+1;
        //Find species solution vector locations for O, O2, OH, CO, H
        //Brute forcing this in here for now
        //Note that since the species vector will always start with Y_C(s) and the dependent variable will always start with temperature, the gas phase species will share the same index in the dependent variable
        // as they do in the species vector
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
            else if (species[sp] == "C2H2")
                C2H2Ind = sp;
            else if (species[sp] == "H2")
                H2Ind = sp;

        if( (H2Ind == -1) | (C2H2Ind == -1) | (HInd == -1) | ( OHInd == -1) | (COInd == -1) | (O2Ind == -1) | (OInd == -1) )
            throw std::out_of_range( "To Use the In Built Soot Mechanism, The UnderLying Mechanism must contain these species: C2H2, H2, H, OH, CO, O2, O!");
    }

    template<typename device_type>
    KOKKOS_INLINE_FUNCTION
        static void UpdateSourceWithSootMechanismRates(const Tines::value_type_1d_view<real_type, device_type>& x, const Tines::value_type_1d_view<real_type, device_type>& f, const real_type &densityTotal, KineticModelConstData<device_type> _kmcd){
        real_type SVF = x(YC_sInd)*densityTotal/eos::TChemSoot::solidCarbonDensity;
        real_type O_SVF = 1-SVF;
        //Scale all Gas Source terms initially by their represented volume fraction
        for(int i =0; i < _kmcd.nSpec; i++) {
            f(i+1) *= O_SVF;
        }
        //Zero Out the temperature source since we technically aren't solving it
        f(0) *= O_SVF;
        //        f(0) = 0;

        f(YC_sInd) = 0;
        f(NddInd) = 1000;
    }

    template<typename device_type,typename MemberType>
    KOKKOS_INLINE_FUNCTION
        static void UpdateSourceWithSootMechanismRatesTemperature(MemberType& member, const Tines::value_type_1d_view<real_type, device_type>& f, const real_type &YCarbon, const real_type &Nd,
                                                       TChem::Impl::StateVector<Tines::value_type_1d_view<real_type, device_type>> gas_SV, const real_type &densityTotal, const Tines::value_type_1d_view<real_type, device_type>& cp_Scratch,
                                                      const Tines::value_type_1d_view<real_type, device_type>& hi_Scratch, KineticModelConstData<device_type> _kmcd){
        using Soot = ablate::eos::TChemSoot;
        double Temperature = gas_SV.Temperature();
        double rho_g = gas_SV.Density();
        auto Yi_g = gas_SV.MassFractions();

        //First Step, alter the species source terms to include the presence of Soot
        real_type SVF = YCarbon*densityTotal/eos::TChemSoot::solidCarbonDensity;
        real_type O_SVF_times_dens_frac = (1-SVF)*rho_g/densityTotal;
        //Scale all Gas Source terms initially by their represented volume fraction, and the density fraction
        //All Tchem species sources are, S_i = \dot{\omega_i}^{mol}*MW_{i}/\rho_gas
        //For our case we want \dot{\omega_i}^{mol}*MW_{i} *(1-f_v)
        //so we have to multiply by (1-f_v) and by rho_gas/rho_{tot}
        for(int i =0; i < _kmcd.nSpec; i++) {
            f(i+1) *= O_SVF_times_dens_frac;
        }
        f(YC_sInd) = 0;
        f(NddInd) = 0;

        //Add in the Soot Reaction Sources
        real_type SA_V,C2H2Conc;
        real_type O2Conc,OConc,OHConc;
        real_type NucRate,SGRate,AggRate,O2OxRate,OOxRate,OHOxRate;
        // Total S.A. of soot / unit volume
        SA_V = calculateSurfaceArea_V(YCarbon,Nd,densityTotal);
        //Need the Concentrations of C2H2, O2, O, and OH
        //It is unclear in the formulations of the Reaction Rates whether to use to concentration in regards to the total mixture or just the gas phace, There is a difference due to the density relation
        //-> For now we will use the concentration to be the concentration in the gas phase as it makes more physical sense
        C2H2Conc = rho_g*PetscMax(0,Yi_g[C2H2Ind-1])/_kmcd.sMass(C2H2Ind-1); // Note that the indices save are for the x vector and are actually 1 more than their corresponding spot in the gas mixture array
        O2Conc = rho_g*PetscMax(0,Yi_g[O2Ind-1])/_kmcd.sMass(O2Ind-1);
        OConc = rho_g*PetscMax(0,Yi_g[OInd-1])/_kmcd.sMass(OInd-1);
        OHConc = rho_g*PetscMax(0,Yi_g[OHInd-1])/_kmcd.sMass(OHInd-1);

        //Now plug in and solve the Nucleation, Surface Growth, Agglomeration, and Oxidation sources
        NucRate = calculateNucleationReactionRate(Temperature, C2H2Conc,SVF);
        SGRate = calculateSurfaceGrowthReactionRate(Temperature, C2H2Conc, SA_V);

        AggRate = calculateAgglomerationRate( YCarbon, Nd, Temperature, densityTotal);
        O2OxRate = calculateO2OxidationRate ( YCarbon, Nd, O2Conc, densityTotal, Temperature, SA_V);
        OOxRate =  calculateOOxidationRate( OConc, Temperature, SA_V, SVF);
        OHOxRate = calculateOHOxidationRate( OHConc, Temperature, SA_V, SVF);
//        AggRate = 0;
//        O2OxRate = 0;
//        OOxRate =  0;
//        OHOxRate = 0;
//        std::cout << std::endl;
//        std::cout << "Current Temperature is            : " << Temperature << std::endl;
//        std::cout << "THE SA_V is                       : " << SA_V << std::endl;
//        std::cout << "The carbon mass fraction is       : " << YCarbon << std::endl;
//        std::cout << "The Actual Soot Number density is : " << Nd << std::endl;
//        std::cout << "The Nucleation Rate is            : " << NucRate << std::endl;
//        std::cout << "The Surface Growth Rate is        : " << SGRate << std::endl;
//        std::cout << "The Agglommeration Rate is        : " << AggRate << std::endl;
//        std::cout << "The O2 Oxidation Rate is          : " << O2OxRate << std::endl;
//        std::cout << "The O Oxidation Rate is           : " << OOxRate << std::endl;
//        std::cout << "The OH Oxidation Rate is          : " << OHOxRate << std::endl;
//        std::cout << "The C2H2 Concentration is         : " << C2H2Conc  << std::endl;
//        std::cout << "The O2 Concentration is           : " << O2Conc  << std::endl;
//        std::cout << "The O Concentration is            : " << OConc  << std::endl;
//        std::cout << "The OH Concentration is           : " << OHConc  << std::endl;

        //Now Add these rates correctly to the appropriate species sources (solving Yidot, i.e. also have to divide by the total density.
        //Keep in mind all these rates are kmol/m^3, need to convert to kg/m^3 for each appropriate species as well!
        real_type O_totDens = 1./densityTotal;
        //C2H2 (Loss from Nucleation and Surface Growth)
        f(C2H2Ind) += O_totDens * _kmcd.sMass(C2H2Ind-1) * (-NucRate - SGRate);
        //O ( Loss from O Oxidation)
        f(OInd) += O_totDens * _kmcd.sMass(OInd-1) * (-OOxRate);
        //O2 (Loss from O2 Oxidation)
        f(O2Ind) += O_totDens * _kmcd.sMass(O2Ind-1) * (-.5 * O2OxRate);
        //OH ( Loss from OH Oxidation)
        f(OHInd) += O_totDens * _kmcd.sMass(OHInd-1) * (-OHOxRate);
        //CO ( Generation From All Oxidations)
        f(COInd) += O_totDens * _kmcd.sMass(COInd-1) * (OHOxRate + O2OxRate + OOxRate);
        //H2 ( Generation From Nucleation and SG)
        f(H2Ind) += O_totDens * _kmcd.sMass(H2Ind-1) * (NucRate + SGRate);
        //H (Generation from OH Oxidation)
        f(HInd) += O_totDens * _kmcd.sMass(HInd-1) * (OHOxRate);
        //Now Onto The Solid Carbon and Ndd source terms
        //SC ( generation from Surface growth and Nucleation and loss from all oxidation's)
        f(YC_sInd) += O_totDens * ablate::eos::TChemSoot::MWCarbon * (2*(NucRate + SGRate) - O2OxRate -OOxRate - OHOxRate);
        f(NddInd) += O_totDens * (NdNuclationConversionTerm * NucRate - AggRate);
        f(NddInd) /= ablate::eos::TChemSoot::NddScaling;



        //Second Alter the Temperature Source term, summing it up myself here, i.e. ignoring that returned by Tchem
        //First get mean gas molecular weight
        real_type MW_gas = TChem::Impl::MolarWeights<real_type,device_type>::team_invoke(member, Yi_g , _kmcd );//MW of gas mixture

        //Second Calculate the mean gas Cv
        real_type Cv_gas = TChem::Impl::CpMixMs<real_type,device_type>::team_invoke(member, Temperature, Yi_g , cp_Scratch, _kmcd) - _kmcd.Runiv/MW_gas;

        //Finally Last thing we need are the species internal energies at this Temperature
        const Tines::value_type_1d_view<real_type, device_type> hi_gas = Kokkos::subview(hi_Scratch,std::make_pair(1,_kmcd.nSpec+1));
        TChem::Impl::EnthalpySpecMlFcn<real_type, device_type>::team_invoke(member, Temperature, hi_gas, cp_Scratch, _kmcd);
        member.team_barrier();

        //Convert Cp_Scratch variable to be the variable holding the gas internal energies
        Kokkos::parallel_for
        (Kokkos::TeamVectorRange(member, _kmcd.nSpec),[&](const ordinal_type& i) {
            cp_Scratch(i) = 1/_kmcd.sMass(i)*(hi_gas(i)-_kmcd.Runiv*Temperature);
           });
        member.team_barrier();

        //The temperature equations is a behemoth,
        //First Grab Some carbon Properties at this temperature
        double HC_R_MWC = Soot::CarbonEnthalpy_R_T(Temperature)*Temperature/Soot::MWCarbon;
        real_type denom = Cv_gas*(1-YCarbon) + YCarbon*_kmcd.Runiv*( HC_R_MWC -
                                                                            rho_g/Soot::solidCarbonDensity/MW_gas);
        real_type YCD = f(_kmcd.nSpec+1);
        //Start the RHS with the Carbon component
        real_type RHS = (_kmcd.Runiv * YCarbon * Temperature /MW_gas / Soot::solidCarbonDensity * YCD * (1/densityTotal-1/Soot::solidCarbonDensity)
                                                                                            / (1/densityTotal-YCarbon/Soot::solidCarbonDensity) / (1/densityTotal-YCarbon/Soot::solidCarbonDensity) );
//        std::cout << "YC: " << YCarbon << ", YCD: " << YCD << ", Rho_T: " << densityTotal;
        //Now Add in the summation terms
        real_type T1 = 0; real_type T2 = 0;
        Kokkos::parallel_for
            (Kokkos::TeamVectorRange(member, _kmcd.nSpec),[&](const ordinal_type& i) {
                //First the normal energy source term
                T1 += - f(i+1)*cp_Scratch(i);
                //Second the Summation Term that comes from the Carbon total energy term (\dot{Y_{i}^T} = (1-Y_C^T) \dot{Y_{i}^g} - \dot{Y_C} Y_{i}^g )
                T2 += ( f(i+1) + Yi_g(i)*YCD ) / (1-YCarbon) / _kmcd.sMass(i) ;
        });
//        std::cout <<", T1: " << T1 << ", T2: " << T2 << ", denom: " << denom << "RHS: " << RHS;
        member.team_barrier();

        //Scale T2 Correctly
        T2 *= YCarbon*_kmcd.Runiv*rho_g*Temperature/Soot::solidCarbonDensity;

        //Add Correct Soot Term To T1
        double sootIntEnergy = HC_R_MWC * _kmcd.Runiv - gas_SV.Pressure()/Soot::solidCarbonDensity;
        T1 += - f(YC_sInd) * sootIntEnergy;
        RHS += T1 + T2;
        f(0) = RHS/denom;
    }

//        void UpdateSourceWithSootMechanismRates(){}

} //End TchemSoot namespace
#endif
