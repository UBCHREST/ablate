#include <filesystem>
#include "MpiTestFixture.hpp"
#include "gtest/gtest.h"
#include "runners/runners.hpp"

INSTANTIATE_TEST_SUITE_P(
    CompressibleFlow, IntegrationTestsSpecifier,
    testing::Values(
        (MpiTestParameter){
            .testName = "inputs/compressibleFlow/compressibleCouetteFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlow/compressibleCouetteFlow.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlow/compressibleFlowVortex.yaml",
                           .nproc = 1,
                           .expectedOutputFile = "outputs/compressibleFlow/compressibleFlowVortex.txt",
                           .arguments = "",
                           .expectedFiles{{"outputs/compressibleFlow/compressibleFlowVortex/domain.xmf", "domain.xmf"}}},
        (MpiTestParameter){
            .testName = "inputs/compressibleFlow/customCouetteCompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlow/customCouetteCompressibleFlow.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlow/extraVariableTransport.yaml",
                           .nproc = 1,
                           .expectedOutputFile = "outputs/compressibleFlow/extraVariableTransport/extraVariableTransport.txt",
                           .arguments = "",
                           .expectedFiles{{"outputs/compressibleFlow/extraVariableTransport/a.csv", "a.csv"},
                                          {"outputs/compressibleFlow/extraVariableTransport/b.csv", "b.csv"},
                                          {"outputs/compressibleFlow/extraVariableTransport/c.csv", "c.csv"},
                                          {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.txt", "rakeProbe/rakeProbe.txt"},
                                          {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.0.csv", "rakeProbe/rakeProbe.0.csv"},
                                          {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.1.csv", "rakeProbe/rakeProbe.1.csv"},
                                          {"outputs/compressibleFlow/extraVariableTransport/rakeProbe/rakeProbe.2.csv", "rakeProbe/rakeProbe.2.csv"}}},
        (MpiTestParameter){.testName = "inputs/compressibleFlow/steadyCompressibleFlowLodiTest.yaml",
                           .nproc = 2,
                           .expectedOutputFile = "outputs/compressibleFlow/steadyCompressibleFlowLodiTest.txt",
                           .arguments = ""},
        (MpiTestParameter){
            .testName = "inputs/compressibleFlow/compressibleFlowVortexLodi.yaml", .nproc = 2, .expectedOutputFile = "outputs/compressibleFlow/compressibleFlowVortexLodi.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlow/compressibleSublimationPipe.yaml",
                           .nproc = 2,
                           .expectedOutputFile = "outputs/compressibleFlow/compressibleSublimationPipe/compressibleSublimationPipe.txt",
                           .arguments = ""},
        (MpiTestParameter){.testName = "inputs/compressibleFlow/compressibleSublimationPipeWithExtrude.yaml",
                           .nproc = 2,
                           .expectedOutputFile = "outputs/compressibleFlow/compressibleSublimationPipeWithExtrude/compressibleSublimationPipeWithExtrude.txt",
                           .arguments = ""},
        (MpiTestParameter){
            .testName = "inputs/compressibleFlow/compressibleFlowCadExample.yaml", .nproc = 1, .expectedOutputFile = "outputs/compressibleFlow/compressibleFlowCadExample.txt", .arguments = ""}),

    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(FEFlow, IntegrationTestsSpecifier,
                         testing::Values(

                             (MpiTestParameter){.testName = "inputs/feFlow/incompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/feFlow/incompressibleFlow.txt", .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Particles, IntegrationTestsSpecifier,
                         testing::Values(

                             (MpiTestParameter){.testName = "inputs/particles/tracerParticles2DHDF5Monitor.yaml",
                                                .nproc = 2,
                                                .expectedOutputFile = "outputs/particles/tracerParticles2DHDF5Monitor.txt",
                                                .arguments = "",
                                                .expectedFiles{{"outputs/particles/tracerParticles2DHDF5Monitor/flowTracerParticles.xmf", "flowTracerParticles.xmf"},
                                                               {"outputs/particles/tracerParticles2DHDF5Monitor/domain.xmf", "domain.xmf"}}},
                             (MpiTestParameter){.testName = "inputs/particles/tracerParticles3D.yaml", .nproc = 1, .expectedOutputFile = "outputs/particles/tracerParticles3D.txt", .arguments = ""},
                             (MpiTestParameter){
                                 .testName = "inputs/particles/inertialParticles2D.yaml", .nproc = 1, .expectedOutputFile = "outputs/particles/inertialParticles2D.txt", .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    VolumeOfFluids, IntegrationTestsSpecifier,
    testing::Values(

        (MpiTestParameter){
            .testName = "inputs/volumeOfFluids/twoGasAdvectingDiscontinuity.yaml", .nproc = 1, .expectedOutputFile = "outputs/volumeOfFluids/twoGasAdvectingDiscontinuity.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/volumeOfFluids/waterGravity.yaml", .nproc = 1, .expectedOutputFile = "outputs/volumeOfFluids/waterGravity.txt", .arguments = ""}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(ReactingFlow, IntegrationTestsSpecifier,
                         testing::Values(

                             (MpiTestParameter){
                                 .testName = "inputs/reactingFlow/simpleReactingFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/reactingFlow/simpleReactingFlow.txt", .arguments = ""},
                             (MpiTestParameter){.testName = "inputs/reactingFlow/ignitionDelayGriMech.yaml",
                                                .nproc = 1,
                                                .arguments = "",
                                                .expectedFiles{{"outputs/reactingFlow/ignitionDelayGriMech.PeakYi.txt", "ignitionDelayPeakYi.txt"}}},
                             (MpiTestParameter){.testName = "inputs/reactingFlow/ignitionDelay2S_CH4_CM2.yaml",
                                                .nproc = 1,
                                                .arguments = "",
                                                .expectedFiles{{"outputs/reactingFlow/ignitionDelay2S_CH4_CM2.Temperature.txt", "ignitionDelayTemperature.txt"}}}),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    Machinery, IntegrationTestsSpecifier,
    testing::Values((MpiTestParameter){.testName = "inputs/machinery/dmViewFromOptions.yaml", .nproc = 1, .expectedOutputFile = "outputs/machinery/dmViewFromOptions.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/machinery/extrudeBoundaryTest.yaml", .nproc = 1, .expectedOutputFile = "outputs/machinery/extrudeBoundaryTest.txt", .arguments = ""},
                    (MpiTestParameter){.testName = "inputs/machinery/subDomainFVM.yaml",
                                       .nproc = 1,
                                       .expectedOutputFile = "outputs/machinery/subDomainFVM/subDomainFVM.txt",
                                       .arguments = "",
                                       .expectedFiles{{"outputs/machinery/subDomainFVM/fluidField.xmf", "fluidField.xmf"}}},
                    (MpiTestParameter){.testName = "inputs/machinery/boundaryMonitorTest2D.yaml",
                                       .nproc = 2,
                                       .expectedOutputFile = "outputs/machinery/boundaryMonitorTest2D/boundaryMonitorTest.txt",
                                       .arguments = "",
                                       .expectedFiles{{"outputs/machinery/boundaryMonitorTest2D/bottomBoundary_monitor.xmf", "bottomBoundary_monitor.xmf"}}},
                    (MpiTestParameter){.testName = "inputs/machinery/boundaryMonitorTest3D.yaml",
                                       .nproc = 1,
                                       .expectedOutputFile = "outputs/machinery/boundaryMonitorTest3D/boundaryMonitorTest.txt",
                                       .arguments = "",
                                       .expectedFiles{{"outputs/machinery/boundaryMonitorTest3D/bottomBoundary_monitor.xmf", "bottomBoundary_monitor.xmf"}}}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(
    ShockTube, IntegrationTestsSpecifier,
    testing::Values(
        (MpiTestParameter){.testName = "inputs/shocktube/shockTubeSODLodiBoundary.yaml", .nproc = 1, .expectedOutputFile = "outputs/shocktube/shockTubeSODLodiBoundary.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/shocktube/shockTube2Gas2D.yaml", .nproc = 1, .expectedOutputFile = "outputs/shocktube/shockTube2Gas2D.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/shocktube/shockTubeRieman.yaml", .nproc = 1, .expectedOutputFile = "outputs/shocktube/shockTubeRieman.txt", .arguments = ""},
        (MpiTestParameter){.testName = "inputs/shocktube/shockTube1DSod_AirWater.yaml", .nproc = 1, .expectedOutputFile = "outputs/shocktube/shockTube1DSod_AirWater.txt", .arguments = ""}),
    [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(CompressibleFlowRestart, IntegrationRestartTestsSpecifier,
                         testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = {.testName = "inputs/compressibleFlow/compressibleFlowPgsLodi.yaml",
                                                                                                  .nproc = 2,
                                                                                                  .expectedOutputFile = "outputs/compressibleFlow/compressibleFlowPgsLodi.txt",
                                                                                                  .arguments = ""},
                                                                             .restartOverrides = {{"timestepper::arguments::ts_max_steps", "50"}}}),
                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) {
                             return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc);
                         });

INSTANTIATE_TEST_SUITE_P(
    RestartFEFlow, IntegrationRestartTestsSpecifier,
    testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = {.testName = "inputs/feFlow/incompressibleFlowRestart.yaml",
                                                                             .nproc = 1,
                                                                             .expectedOutputFile = "outputs/feFlow/incompressibleFlowRestart.txt",
                                                                             .arguments = "",
                                                                             .expectedFiles = {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}},
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
                    (IntegrationRestartTestsParameters){.mpiTestParameter = {.testName = "inputs/feFlow/incompressibleFlowRestart.yaml",
                                                                             .nproc = 2,
                                                                             .expectedOutputFile = "outputs/feFlow/incompressibleFlowRestart.txt",
                                                                             .arguments = "",
                                                                             .expectedFiles = {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}},
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
                    (IntegrationRestartTestsParameters){.mpiTestParameter = {.testName = "inputs/feFlow/incompressibleFlowIntervalRestart.yaml",
                                                                             .nproc = 1,
                                                                             .expectedOutputFile = "outputs/feFlow/incompressibleFlowIntervalRestart.txt",
                                                                             .arguments = ""},
                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });

INSTANTIATE_TEST_SUITE_P(FVRestart, IntegrationRestartTestsSpecifier,
                         testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = {.testName = "inputs/compressibleFlow/compressibleFlowVortexLodiRestart.yaml",
                                                                                                  .nproc = 1,
                                                                                                  .expectedOutputFile = "outputs/compressibleFlow/compressibleFlowVortexLodiRestart.txt",
                                                                                                  .arguments = ""},
                                                                             .restartOverrides = {{"timestepper::arguments::ts_max_steps", "20"}}}),
                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) {
                             return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc);
                         });

INSTANTIATE_TEST_SUITE_P(
    RestartParticles, IntegrationRestartTestsSpecifier,
    testing::Values((IntegrationRestartTestsParameters){
        .mpiTestParameter = {.testName = "inputs/particles/tracerParticles2DRestart.yaml", .nproc = 1, .expectedOutputFile = "outputs/particles/tracerParticles2DRestart.txt", .arguments = ""},
        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });

INSTANTIATE_TEST_SUITE_P(Monitors, IntegrationTestsSpecifier,
                         testing::Values((MpiTestParameter){.testName = "inputs/monitors/rocketMonitor.yaml", .nproc = 1, .expectedOutputFile = "outputs/monitors/rocketMonitor.txt", .arguments = ""},
                                         (MpiTestParameter){.testName = "inputs/monitors/turbFlowStatsMonitor.yaml",
                                                            .nproc = 1,
                                                            .arguments = "",
                                                            .expectedFiles{{"outputs/monitors/turbFlowStatsMonitor/TurbFlowStats.xmf", "TurbFlowStats.xmf"},
                                                                           {"outputs/monitors/turbFlowStatsMonitor/domain.xmf", "domain.xmf"}}}),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });

INSTANTIATE_TEST_SUITE_P(Radiation, IntegrationTestsSpecifier,
                         testing::Values(

                             (MpiTestParameter){
                                 .testName = "inputs/radiation/parallelPlatesRadiation.yaml", .nproc = 1, .expectedOutputFile = "outputs/radiation/parallelPlatesOutput.txt", .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });
