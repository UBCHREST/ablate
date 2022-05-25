#ifndef ABLATELIBRARY_SENSIBLEINTERNALENERGY_HPP
#define ABLATELIBRARY_SENSIBLEINTERNALENERGY_HPP

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace ablate::eos::tChem {

class SensibleInternalEnergy {
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type, host_device_type>;
    using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type, host_device_type>;

    using kinetic_model_type = KineticModelConstData<device_type>;
    using kinetic_model_host_type = KineticModelConstData<host_device_type>;

    static inline ordinal_type getWorkSpaceSize() {
        return 0;  // does not need work size
    }

    /**
     * tchem like function to compute temperature on device
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param mwMix
     * @param temperature
     * @param kmcd
     */
    static void runDeviceBatch(  /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy,
        /// the output is the updated temperature in the state
        const real_type_2d_view_type& state, const real_type_1d_view_type& internalEnergyRef, const real_type_1d_view_type& mwMix,
        /// const data from kinetic model
        const kinetic_model_type& kmcd);

    /**
     * tchem like function to compute temperature on host
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param mwMix
     * @param temperature
     * @param kmcd
     */
    static void runHostBatch(  /// thread block size
        typename UseThisTeamPolicy<host_exec_space>::type& policy,
        /// the output is the updated temperature in the state
        const real_type_2d_view_host_type& state, const real_type_1d_view_host_type& internalEnergyRef, const real_type_1d_view_host_type& mwMix,
        /// const data from kinetic model
        const kinetic_model_type& kmcd);
};

}  // namespace ablate::eos::tChem
#endif  // ABLATELIBRARY_TCHEMTEMPERATURE_HPP
