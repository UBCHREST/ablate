#ifndef ABLATELIBRARY_TOTALINTERNALENERGYFCN_HPP
#define ABLATELIBRARY_TOTALINTERNALENERGYFCN_HPP

#include <TChem_Impl_EnthalpySpecMl.hpp>
#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"
#include "eos/tChem/pressureFcn.hpp"
#include "eos/tChemSoot.hpp"

namespace tChemLib = TChem;

namespace ablate::eos::tChemSoot::impl {

template <typename ValueType, typename DeviceType>
struct TotalInternalEnergyFcn {
    using value_type = ValueType;
    using device_type = DeviceType;
    using scalar_type = typename ats<value_type>::scalar_type;

    using real_type = scalar_type;
    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    /// sacado is value type
    using value_type_1d_view_type = Tines::value_type_1d_view<value_type, device_type>;
    using kinetic_model_type = KineticModelConstData<device_type>;

    template <typename MemberType, typename KineticModelConstDataType>
    KOKKOS_INLINE_FUNCTION static value_type team_invoke(const MemberType& member,
                                                         /// YCarbon
                                                         const value_type& YCarbon,
                                                         ///Gaseous StateVector
                                                         const Impl::StateVector<value_type_1d_view_type> sv_gas,
                                                         ///All Species Internal enthalpies
                                                         const value_type_1d_view_type& hi,
                                                         ///Should be sized for kmcd Species
                                                         const value_type_1d_view_type& cpks,
                                                         /// Constant Kinetics Model
                                                         const KineticModelConstDataType& kmcd) {
        // compute the enthalpy of each species at temperature
        value_type temperature = sv_gas.Temperature();
        auto ys_Gas = sv_gas.MassFractions();
        //Calculate the Pressure of the Mixture
        auto pressure = ablate::eos::tChem::impl::pressureFcn<real_type, device_type>::team_invoke(member, sv_gas, kmcd); //Needed for the soot internal energy
        //Grab the species specific enthalpies in the gas
        const real_type_1d_view_type hi_gas = Kokkos::subview(hi,std::make_pair(1,kmcd.nSpec+1)); //hi array for all gaseous species vectors
        tChemLib::Impl::EnthalpySpecMlFcn<value_type, device_type>::team_invoke(member, temperature, hi_gas, cpks, kmcd);
        member.team_barrier();
        // compute the sensibleInternalEnergy (Gaseous)
        value_type totalInternalEnergy;
        Kokkos::parallel_reduce(
            Kokkos::TeamVectorRange(member, kmcd.nSpec),
            [&](const ordinal_type& k, real_type& update) { update += (hi_gas(k) / kmcd.sMass(k) - kmcd.Runiv * temperature / kmcd.sMass(k)) * ys_Gas(k); },
            totalInternalEnergy);
        //Scale Internal Energy with the solid component
        totalInternalEnergy*= (1-YCarbon);
        //Add on the contribution due to soot //e_{sens,carbon} = H_i_mass - P/rhoCarbon
        value_type sootInternalEnergy = kmcd.Runiv*temperature*ablate::eos::TChemSoot::CarbonEnthalpy_R_T(temperature)/ablate::eos::TChemSoot::MWCarbon - pressure/ablate::eos::TChemSoot::solidCarbonDensity;
        totalInternalEnergy += sootInternalEnergy*YCarbon;
        return totalInternalEnergy;
    }
};


}  // namespace ablate::eos::tChem::impl
#endif
