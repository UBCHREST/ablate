//
// Created by klbud on 10/3/22.
//

#ifndef ABLATE_SPECIFICHEATCONSTANTVOLUME_HPP
#define ABLATE_SPECIFICHEATCONSTANTVOLUME_HPP
#include <Kokkos_Core.hpp>
#ifndef KOKKOS_ENABLE_CUDA

#include "TChem_KineticModelData.hpp"
#include "TChem_Util.hpp"

namespace ablate::eos::tChemSoot {

struct SpecificHeatConstantVolume {
    using host_device_type = typename Tines::UseThisDevice<host_exec_space>::type;
    using device_type = typename Tines::UseThisDevice<exec_space>::type;

    using real_type_1d_view_type = Tines::value_type_1d_view<real_type, device_type>;
    using real_type_2d_view_type = Tines::value_type_2d_view<real_type, device_type>;

    using real_type_1d_view_host_type = Tines::value_type_1d_view<real_type, host_device_type>;
    using real_type_2d_view_host_type = Tines::value_type_2d_view<real_type, host_device_type>;

    using kinetic_model_type = KineticModelConstData<device_type>;
    using kinetic_model_host_type = KineticModelConstData<host_device_type>;

    static inline ordinal_type getWorkSpaceSize(ordinal_type numberSpecies) { return numberSpecies; }

    /**
     * tchem like function to compute temperature on device
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param mwMix
     * @param temperature
     * @param kmcd
     */
    [[maybe_unused]] static void runDeviceBatch(  /// thread block size
        typename UseThisTeamPolicy<exec_space>::type& policy,
        /// state
        const real_type_2d_view_type& state,
        /// output
        const real_type_1d_view_type& CvMix,
        /// const data from kinetic model
        const kinetic_model_type& kmcd);

    /**
     * tchem like function to compute temperature on host
const real_type_2d_view_host_type& state,
     * @param policy
     * @param state
     * @param internalEnergyRef
     * @param mwMix
     * @param temperature
     * @param kmcd
     */
    [[maybe_unused]] static void runHostBatch(  /// thread block size
        typename UseThisTeamPolicy<host_exec_space>::type& policy,
        /// input
        const real_type_2d_view_host_type& state,
        /// output
        const real_type_1d_view_host_type& CvMix,
        /// const data from kinetic model
        const kinetic_model_host_type& kmcd);
};

}  // namespace ablate::eos::tChemSoot
#endif  // ABLATE_SPECIFICHEATCONSTANTVOLUME_HPP
#endif