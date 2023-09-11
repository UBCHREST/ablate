//#include <filesystem>
//#include "MpiTestFixture.hpp"
//#include "gtest/gtest.h"
//#include "runners/runners.hpp"
//
//INSTANTIATE_TEST_SUITE_P(
//    CompressibleFlow, IntegrationTestsSpecifier,
//    testing::Values(
//



//INSTANTIATE_TEST_SUITE_P(
//    RestartFEFlow, IntegrationRestartTestsSpecifier,
//    testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowRestart.yaml", 1, "", "outputs/feFlow/incompressibleFlowRestart.txt",
//                                                                                             {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}),
//                                                        .restartInputFile = "",
//                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
//                    (IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowRestart.yaml", 2, "", "outputs/feFlow/incompressibleFlowRestart.txt",
//                                                                                             {{"outputs/feFlow/incompressibleFlowRestartProbe.csv", "incompressibleFlowRestartProbe.csv"}}),
//                                                        .restartInputFile = "",
//                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "30"}}},
//                    (IntegrationRestartTestsParameters){
//                        .mpiTestParameter = MpiTestParameter("inputs/feFlow/incompressibleFlowIntervalRestart.yaml", 1, "", "outputs/feFlow/incompressibleFlowIntervalRestart.txt"),
//                        .restartInputFile = "",
//                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
//    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });
//
//INSTANTIATE_TEST_SUITE_P(FVRestart, IntegrationRestartTestsSpecifier,
//                         testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/compressibleFlow/compressibleFlowVortexLodiRestart.yaml", 1, "",
//                                                                                                                  "outputs/compressibleFlow/compressibleFlowVortexLodiRestart.txt"),
//                                                                             .restartInputFile = "",
//                                                                             .restartOverrides = {{"timestepper::arguments::ts_max_steps", "20"}}}),
//                         [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) {
//                             return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc);
//                         });
//
//INSTANTIATE_TEST_SUITE_P(
//    RestartParticles, IntegrationRestartTestsSpecifier,
//    testing::Values((IntegrationRestartTestsParameters){.mpiTestParameter = MpiTestParameter("inputs/particles/tracerParticles2DRestart.yaml", 1, "", "outputs/particles/tracerParticles2DRestart.txt"),
//                                                        .restartInputFile = "",
//                                                        .restartOverrides = {{"timestepper::arguments::ts_max_steps", "10"}}}),
//    [](const testing::TestParamInfo<IntegrationRestartTestsParameters>& info) { return info.param.mpiTestParameter.getTestName() + "_" + std::to_string(info.param.mpiTestParameter.nproc); });
//
//INSTANTIATE_TEST_SUITE_P(Monitors, IntegrationTestsSpecifier,
//                         testing::Values(MpiTestParameter("inputs/monitors/rocketMonitor.yaml", 1, "", "outputs/monitors/rocketMonitor.txt"),
//
//
//                                         MpiTestParameter("inputs/monitors/turbFlowStatsMonitor.yaml", 1, "", "",
//                                                          {{"outputs/monitors/turbFlowStatsMonitor/flowField_turbulenceFlowStats.xmf", "flowField_turbulenceFlowStats.xmf"},
//                                                           {"outputs/monitors/turbFlowStatsMonitor/domain.xmf", "domain.xmf"}}),



//                                         MpiTestParameter("inputs/monitors/radiationFieldMonitor.yaml", 1, "", "",
//                                                          {{"outputs/monitors/radiationFieldMonitor/radiationFieldMonitor.xmf", "radiationFieldMonitor.xmf"},
//                                                           {"outputs/monitors/radiationFieldMonitor/domain.xmf", "domain.xmf"}}),
//                                         MpiTestParameter("inputs/monitors/radiationFlux.yaml", 1, "", "",
//                                                          {{"outputs/monitors/radiationFlux/upperWallBoundaryFaces_radiationFluxMonitor.xmf", "upperWallBoundaryFaces_radiationFluxMonitor.xmf"},
//                                                           {"outputs/monitors/radiationFlux/domain.xmf", "domain.xmf"}})),
//                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });
//
//INSTANTIATE_TEST_SUITE_P(Radiation, IntegrationTestsSpecifier,
//                         testing::Values(MpiTestParameter("inputs/radiation/parallelPlatesRadiation.yaml", 1, "", "outputs/radiation/parallelPlatesOutput.txt"),
//                                         MpiTestParameter("inputs/radiation/virtualTCP.yaml", 1, "", "outputs/radiation/virtualTCP.txt")),
//                         [](const testing::TestParamInfo<MpiTestParameter>& info) { return info.param.getTestName(); });
