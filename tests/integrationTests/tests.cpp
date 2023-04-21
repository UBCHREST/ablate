#include <filesystem>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "runners/runners.hpp"

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, IntegrationTestsSpecifier,
    testing::Values(MpiTestParameter("inputs/compressibleFlow/compressibleCouetteFlow.yaml", 1, "", "outputs/compressibleFlow/compressibleCouetteFlow.txt"),
                    MpiTestParameter("inputs/compressibleFlow/compressibleFlowVortex.yaml", 1, "", "outputs/compressibleFlow/compressibleFlowVortex.txt",
                                     {{"outputs/compressibleFlow/compressibleFlowVortex/domain.xmf", "domain.xmf"}}),
                    MpiTestParameter("inputs/compressibleFlow/turbulentFlow/lesIsotropicTurbulence.yaml", 1, "", "outputs/compressibleFlow/lesIsotropicTurbulence.txt",
                                     {{"outputs/compressibleFlow/lesIsotropicTurbulence/domain.xmf", "domain.xmf"}}),
                    MpiTestParameter("inputs/compressibleFlow/turbulentFlow/turbulentChannelFlow.yaml", 1, "", "", {{"outputs/compressibleFlow/turbulentChannelFlow/domain.xmf", "domain.xmf"}}),
                    MpiTestParameter("inputs/compressibleFlow/customCouetteCompressibleFlow.yaml", 1, "outputs/compressibleFlow/customCouetteCompressibleFlow.txt", ""),
                    MpiTestParameter("inputs/compressibleFlow/extraVariableTransport.yaml", 1, "", "outputs/compressibleFlow/extraVariableTransport/extraVariableTransport.txt",
                                     {{"outputs/compressibleFlow/extraVariableTransport/a.csv", "a.csv"},
                                      {"outputs/compressibleFlow/extraVariableTransport/b.csv", "b.csv"},
                                      {"outputs/compressibleFlow/extraVariableTransport/c.csv", "c.csv"},
                                      {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.txt", "rakeProbe/rakeProbe.txt"},
                                      {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.0.csv", "rakeProbe/rakeProbe.0.csv"},
                                      {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.1.csv", "rakeProbe/rakeProbe.1.csv"},
                                      {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.2.csv", "rakeProbe/rakeProbe.2.csv"}}),
                    MpiTestParameter("inputs/compressibleFlow/steadyCompressibleFlowLodiTest.yaml", 2, "", "outputs/compressibleFlow/steadyCompressibleFlowLodiTest.txt"),
                    MpiTestParameter("inputs/compressibleFlow/compressibleFlowVortexLodi.yaml", 2, "outputs/compressibleFlow/compressibleFlowVortexLodi.txt", ""),
                    MpiTestParameter("inputs/compressibleFlow/compressibleSublimationPipe.yaml", 2, "", "outputs/compressibleFlow/compressibleSublimationPipe/compressibleSublimationPipe.txt"),
                    MpiTestParameter("inputs/compressibleFlow/compressibleSublimationPipeWithExtrude.yaml", 2, "",
                                     "outputs/compressibleFlow/compressibleSublimationPipeWithExtrude/compressibleSublimationPipeWithExtrude.txt"),
                    MpiTestParameter("inputs/compressibleFlow/gmshPipeFlow/gmshPipeFlow.yaml", 2, "", "outputs/compressibleFlow/gmshPipeFlow/gmshPipeFlow.txt"),
                    MpiTestParameter("inputs/compressibleFlow/compressibleFlowCadExample.yaml", 1, "", "outputs/compressibleFlow/compressibleFlowCadExample.txt", {}, "ASAN_OPTIONS=detect_leaks=0")),

    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(FEFlow, IntegrationTestsSpecifier, testing::Values(MpiTestParameter("inputs/feFlow/incompressibleFlow.yaml", 1, "", "outputs/feFlow/incompressibleFlow.txt")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Particles, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/particles/tracerParticles2DHDF5Monitor.yaml", 2, "", "outputs/particles/tracerParticles2DHDF5Monitor.txt",
                                                          {{"outputs/particles/tracerParticles2DHDF5Monitor/flowTracerParticles.xmf", "flowTracerParticles.xmf"},
                                                           {"outputs/particles/tracerParticles2DHDF5Monitor/domain.xmf", "domain.xmf"}}),
                                         MpiTestParameter("inputs/particles/tracerParticles3D.yaml", 1, "", "outputs/particles/tracerParticles3D.txt"),
                                         MpiTestParameter("inputs/particles/inertialParticles2D.yaml", 1, "", "outputs/particles/inertialParticles2D.txt")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(VolumeOfFluids, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/volumeOfFluids/twoGasAdvectingDiscontinuity.yaml", 1, "", "outputs/volumeOfFluids/twoGasAdvectingDiscontinuity.txt"),
                                         MpiTestParameter("inputs/volumeOfFluids/twoPhaseCouetteFlow.yaml", 1, "", "outputs/volumeOfFluids/twoPhaseCouetteFlow.txt"),
                                         MpiTestParameter("inputs/volumeOfFluids/waterGravity.yaml", 1, "", "outputs/volumeOfFluids/waterGravity.txt")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    ReactingFlow, IntegrationTestsSpecifier,
    testing::Values(

        MpiTestParameter("inputs/reactingFlow/simpleReactingFlow.yaml", 1, "", "outputs/reactingFlow/simpleReactingFlow.txt"),
        MpiTestParameter("inputs/reactingFlow/sampleDiffusionFlame.yaml", 1, "", "outputs/reactingFlow/sampleDiffusionFlame.txt"),
        MpiTestParameter("inputs/reactingFlow/sampleSootDiffusionFlame.yaml", 1, "", "outputs/reactingFlow/sampleSootDiffusionFlame.txt"),
        MpiTestParameter("inputs/reactingFlow/ignitionDelayGriMech.yaml", 1, "", "",
                         {{"outputs/reactingFlow/ignitionDelayGriMech.PeakYi.txt", "ignitionDelayPeakYi.txt"},
                          {"outputs/reactingFlow/ignitionDelayGriMech.Temperature.txt", "ignitionDelayTemperature.csv"}}),
        MpiTestParameter("inputs/reactingFlow/ignitionDelay2S_CH4_CM2.yaml", 1, "", "", {{"outputs/reactingFlow/ignitionDelay2S_CH4_CM2.Temperature.txt", "ignitionDelayTemperature.txt"}}),
        MpiTestParameter("inputs/reactingFlow/ignitionDelayMMASoot.yaml", 1, "", "", {{"outputs/reactingFlow/ignitionDelayMMASoot.Temperature.txt", "ignitionDelayTemperature.csv"}}),
        MpiTestParameter("inputs/reactingFlow/ignitionDelayMMASootProcess.yaml", 1, "", "", {{"outputs/reactingFlow/ignitionDelayMMASootProcess.Temperature.txt", "ignitionDelayTemperature.csv"}})),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Machinery, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/machinery/dmViewFromOptions.yaml", 1, "", "outputs/machinery/dmViewFromOptions.txt"),
                                         MpiTestParameter("inputs/machinery/extrudeBoundaryTest.yaml", 1, "", "outputs/machinery/extrudeBoundaryTest.txt"),
                                         MpiTestParameter("inputs/machinery/meshMappingTest.yaml", 1, "", "outputs/machinery/meshMappingTest.txt"),
                                         MpiTestParameter("inputs/machinery/meshMappingTestCoordinateSpace.yaml", 1, "", "outputs/machinery/meshMappingTestCoordinateSpace.txt"),
                                         MpiTestParameter("inputs/machinery/subDomainFVM.yaml", 1, "", "outputs/machinery/subDomainFVM/subDomainFVM.txt",
                                                          {{"outputs/machinery/subDomainFVM/fluidField.xmf", "fluidField.xmf"}}),
                                         MpiTestParameter("inputs/machinery/boundaryMonitorTest2D.yaml", 2, "", "outputs/machinery/boundaryMonitorTest2D/boundaryMonitorTest2D.txt",
                                                          {{"outputs/machinery/boundaryMonitorTest2D/bottomBoundary_monitor.xmf", "bottomBoundary_monitor.xmf"}}),
                                         MpiTestParameter("inputs/machinery/boundaryMonitorTest3D.yaml", 1, "", "outputs/machinery/boundaryMonitorTest3D/boundaryMonitorTest3D.txt",
                                                          {{"outputs/machinery/boundaryMonitorTest3D/bottomBoundary_monitor.xmf", "bottomBoundary_monitor.xmf"}})),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(ShockTube, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/shocktube/shockTubeSODLodiBoundary.yaml", 1, "", "outputs/shocktube/shockTubeSODLodiBoundary.txt"),
                                         MpiTestParameter("inputs/shocktube/shockTube2Gas2D.yaml", 1, "", "outputs/shocktube/shockTube2Gas2D.txt"),
                                         MpiTestParameter("inputs/shocktube/shockTubeRieman.yaml", 1, "", "outputs/shocktube/shockTubeRieman.txt"),
                                         MpiTestParameter("inputs/shocktube/shockTube1DSod_AirWater.yaml", 1, "", "outputs/shocktube/shockTube1DSod_AirWater.txt")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(CompressibleFlowRestart, IntegrationRestartTestsSpecifier,
                         testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/compressibleFlow/compressibleFlowPgsLodi.yaml", 2, "",
                                                                                                                  "outputs/compressibleFlow/compressibleFlowPgsLodi.txt"),
                                                                             .restartInputFile = "",
                                                                             .restartOverrides = {{"timestepper::arguments::ts_max_steps", "50"}}},

                                         (IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/compressibleFlow/hdf5InitializerTest/hdf5InitializerTest.yaml", 1, "",
                                                                                                                  "outputs/compressibleFlow/hdf5InitializerTest/hdf5InitializerTest.txt"),
                                                                             .restartInputFile = "inputs/compressibleFlow/hdf5InitializerTest/hdf5InitializerTest.Initialization.yaml",
                                                                             .restartOverrides = {}}),
                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) {
                             return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc);
                         });

INSTANTIATE_TEST_SUITE_P(
    RestartFEFlow, IntegrationRestartTestsSpecifier,
    testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowRestart.yaml", 1, "", "outputs/feFlow/incompressibleFlowRestart.txt",
                                                                                             {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}),
                                                        .restartInputFile = "",
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
                    (IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowRestart.yaml", 2, "", "outputs/feFlow/incompressibleFlowRestart.txt",
                                                                                             {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}),
                                                        .restartInputFile = "",
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
                    (IntegrationRestartTestsParameters){
                        .mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowIntervalRestart.yaml", 1, "", "outputs/feFlow/incompressibleFlowIntervalRestart.txt"),
                        .restartInputFile = "",
                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });

INSTANTIATE_TEST_SUITE_P(FVRestart, IntegrationRestartTestsSpecifier,
                         testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/compressibleFlow/compressibleFlowVortexLodiRestart.yaml", 1, "",
                                                                                                                  "outputs/compressibleFlow/compressibleFlowVortexLodiRestart.txt"),
                                                                             .restartInputFile = "",
                                                                             .restartOverrides = {{"timestepper::arguments::ts_max_steps", "20"}}}),
                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) {
                             return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc);
                         });

INSTANTIATE_TEST_SUITE_P(
    RestartParticles, IntegrationRestartTestsSpecifier,
    testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/particles/tracerParticles2DRestart.yaml", 1, "", "outputs/particles/tracerParticles2DRestart.txt"),
                                                        .restartInputFile = "",
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });

INSTANTIATE_TEST_SUITE_P(Monitors, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/monitors/rocketMonitor.yaml", 1, "", "outputs/monitors/rocketMonitor.txt"),
                                         MpiTestParameter("inputs/monitors/turbFlowStatsMonitor.yaml", 1, "", "",
                                                          {{"outputs/monitors/turbFlowStatsMonitor/flowField_turbulenceFlowStats.xmf", "flowField_turbulenceFlowStats.xmf"},
                                                           {"outputs/monitors/turbFlowStatsMonitor/domain.xmf", "domain.xmf"}}),
                                         MpiTestParameter("inputs/monitors/radiationFieldMonitor.yaml", 1, "", "",
                                                          {{"outputs/monitors/radiationFieldMonitor/radiationFieldMonitor.xmf", "radiationFieldMonitor.xmf"},
                                                           {"outputs/monitors/radiationFieldMonitor/domain.xmf", "domain.xmf"}}),
                                         MpiTestParameter("inputs/monitors/radiationFlux.yaml", 1, "", "",
                                                          {{"outputs/monitors/radiationFlux/upperWallBoundaryFaces_radiationFluxMonitor.xmf", "upperWallBoundaryFaces_radiationFluxMonitor.xmf"},
                                                           {"outputs/monitors/radiationFlux/domain.xmf", "domain.xmf"}})),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Radiation, IntegrationTestsSpecifier,
                         testing::Values(MpiTestParameter("inputs/radiation/parallelPlatesRadiation.yaml", 1, "", "outputs/radiation/parallelPlatesOutput.txt"),
                                         MpiTestParameter("inputs/radiation/virtualTCP.yaml", 1, "", "outputs/radiation/virtualTCP.txt"),
                                         MpiTestParameter("inputs/radiation/spectrumRadiation.yaml", 1, "", "outputs/radiation/spectrumRadiation.txt")),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });
