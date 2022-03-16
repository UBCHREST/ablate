#include <petsc.h>
#include <vector>
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "finiteVolume/fluxCalculator/ausm.hpp"
#include "finiteVolume/fluxCalculator/ausmpUp.hpp"
#include "finiteVolume/fluxCalculator/averageFlux.hpp"
#include "finiteVolume/fluxCalculator/offFlux.hpp"
#include "finiteVolume/fluxCalculator/rieman.hpp"
#include "finiteVolume/fluxCalculator/riemann2Gas.hpp"
#include "finiteVolume/fluxCalculator/riemannStiff.hpp"
#include "gtest/gtest.h"
#include "parameters/mapParameters.hpp"

using namespace ablate::finiteVolume::fluxCalculator;

struct FluxCalculatorTestParameters {
    std::string testName;
    std::shared_ptr<ablate::finiteVolume::fluxCalculator::FluxCalculator> fluxCalculator;
    std::vector<PetscReal> uL;
    std::vector<PetscReal> aL;
    std::vector<PetscReal> rhoL;
    std::vector<PetscReal> pL;
    std::vector<PetscReal> uR;
    std::vector<PetscReal> aR;
    std::vector<PetscReal> rhoR;
    std::vector<PetscReal> pR;
    std::vector<PetscReal> expectedMassFlux;
    std::vector<PetscReal> expectedInterfacePressure;
    std::vector<ablate::finiteVolume::fluxCalculator::Direction> expectedDirection;
};

class FluxCalculatorTestParametersTestFixture : public ::testing::TestWithParam<FluxCalculatorTestParameters> {};

TEST_P(FluxCalculatorTestParametersTestFixture, ShouldComputeCorrectFlux) {
    // arrange
    const auto& params = GetParam();

    FluxCalculatorFunction function = params.fluxCalculator->GetFluxCalculatorFunction();
    void* context = params.fluxCalculator->GetFluxCalculatorContext();

    PetscReal massFlux;
    PetscReal pressureFace;

    // act
    for (std::size_t i = 0; i < params.uL.size(); i++) {
        auto direction = function(context, params.uL[i], params.aL[i], params.rhoL[i], params.pL[i], params.uR[i], params.aR[i], params.rhoR[i], params.pR[i], &massFlux, &pressureFace);

        // assert
        EXPECT_EQ(direction, params.expectedDirection[i]);
        EXPECT_NEAR(params.expectedMassFlux[i], massFlux, 1E-2);
        EXPECT_NEAR(params.expectedInterfacePressure[i], pressureFace, 1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    FluxDifferencer, FluxCalculatorTestParametersTestFixture,
    testing::Values(
        (FluxCalculatorTestParameters){
            .testName = "Ausm",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Ausm>(),
            .uL = {-46.801376,  72.26622,    143.493328, -298.81582,  -7.23096,    -18.991384,  197.677272,  139.742053,  -594.589692, -290.53794, 428.43614,   -54.055332,  115.4032,
                   -392.340472, -41.287239,  154.19184,  -107.410149, 605.56822,   237.64508,   -322.95408,  -229.684257, 14.9085,     426.853071, -108.341724, -361.619038, -27.726526,
                   -194.035211, -128.021795, -51.42915,  -2.328389,   -207.126428, -170.092372, -109.112972, -17.289702,  -100.064006, -363.66779, 429.399216,  41.324511,   141.155896,
                   -202.276375, 738.81525,   380.103504, -210.78516,  -110.421135, 53.15628,    272.014974,  668.626846,  -23.056328,  369.64004,  51.09165,    -277.643678, 88.631466,
                   281.504544,  465.28944,   1.05675,    33.464846,   -655.387744, -273.72464,  247.440336,  -108.649848, -460.40544,  50.36031,   59.287899,   92.128476,   -147.349272,
                   -94.731364,  8.190558,    238.34678,  26.885034,   -136.6336,   -15.87201,   -178.7615,   -657.37836,  237.550176,  556.632288, 42.119055,   -20.003247,  -374.483252,
                   1.719168,    225.611919,  330.038365, 36.76504,    396.318725,  -347.09958,  7.987615,    246.14232,   48.232745,   -18.799395, -130.126362, 18.622545,   464.43011,
                   -221.442256, 211.039506,  443.552624, 134.390466,  33.67049,    383.3336,    2.764381,    73.383869,   -84.256545},
            .aL = {32,  52,  136, 305, 396, 94,  326, 121, 366, 154, 319, 201, 110, 197, 189, 174, 99,  340, 148, 172, 121, 90, 243, 228, 319, 89,  103, 115, 207, 139, 194, 158, 97,  107,
                   119, 190, 318, 183, 248, 161, 375, 264, 139, 165, 60,  322, 383, 74,  280, 75,  353, 253, 184, 336, 50,  94, 352, 266, 208, 317, 352, 258, 73,  132, 294, 388, 243, 214,
                   321, 320, 385, 95,  360, 292, 288, 113, 47,  284, 192, 117, 305, 232, 275, 324, 145, 136, 155, 115, 141, 97, 254, 112, 186, 346, 87,  35,  310, 37,  299, 285},
            .rhoL = {1, 3, 5, 6, 2, 6, 3, 2, 1, 2, 6, 5, 7, 6, 5, 4, 2, 6, 4, 6, 6, 5, 3, 7, 4, 5, 1, 4, 5, 7, 2, 7, 3, 1, 4, 3, 5, 5, 2, 5, 4, 6, 4, 2, 2, 4, 5, 4, 1, 3,
                     1, 6, 4, 7, 4, 6, 2, 3, 5, 2, 1, 6, 4, 6, 4, 3, 1, 7, 7, 3, 3, 1, 6, 2, 5, 7, 3, 7, 5, 6, 1, 4, 4, 6, 4, 4, 1, 7, 7, 5, 3, 1, 5, 4, 7, 2, 5, 3, 6, 7},
            .pL = {51426,  264102, 254728, 195151, 172324, 256522, 251864, 388697, 283138, 11038,  66395,  344641, 108220, 35039,  371580, 203112, 134372, 221364, 300401, 138026,
                   211427, 317949, 276653, 377959, 381839, 1551,   245777, 94258,  212125, 58927,  84174,  280084, 127030, 227258, 298628, 73291,  93388,  94826,  192988, 72043,
                   338841, 247224, 278942, 308217, 206011, 261261, 355440, 49699,  184743, 379053, 351617, 387637, 302774, 75536,  136996, 339013, 128304, 295260, 299086, 199620,
                   118599, 272950, 215031, 349146, 174233, 67125,  133440, 224038, 27598,  57909,  242843, 182852, 381089, 190538, 133120, 94895,  135304, 253163, 176757, 289661,
                   348481, 31037,  154440, 274750, 125417, 218854, 88891,  82449,  83827,  397410, 51356,  391009, 104810, 35399,  51356,  391009, 104810, 125417, 218854, 88891},
            .uR = {118.40328,   94.358989,  -322.972698, 3.58758,     -16.1001,    -199.34832, -121.906224, 51.09978,   108.67076,  107.05478,   -57.255246,  -664.6112,   -405.5925,
                   -169.141305, 107.434074, 174.728235,  105.050528,  117.65061,   413.899923, -342.5884,   416.928595, 256.900434, -655.155759, -398.246822, -508.74112,  74.631331,
                   -332.585316, -103.09497, 183.209336,  -68.45399,   -459.757446, -142.95729, 41.727882,   -110.17586, -47.233784, 265.75704,   -54.720286,  -181.558215, -8.288364,
                   -245.804598, -45.830972, -25.306,     -354.387412, -300.470976, 404.616322, 8.942544,    -74.472704, 126.355905, 182.614896,  -264.508464, -352.923186, -330.911802,
                   254.724522,  -426.75426, 213.608403,  51.314616,   32.722047,   -212.34419, 511.739682,  194.172609, 5.97582,    -231.408353, 98.608316,   -66.424464,  86.46444,
                   -119.407839, 53.950473,  -661.8225,   -209.44976,  555.869061,  337.980825, -301.262606, 127.631162, -100.12794, 138.815166,  173.615504,  -74.774353,  -258.538248,
                   -267.386744, 539.294432, -221.80008,  245.309952,  367.912195,  57.348888,  -115.626016, 576.680608, 513.572371, 124.046136,  9.79662,     -187.928125, -86.330916,
                   -183.604668, 55.032348,  -75.6324,    -172.517025, -246.839268, 52.44257,   -115.014969, -0.1248,    -55.190743},
            .aR = {80,  337, 269, 190, 267, 348, 198, 332, 220, 238, 342, 352, 375, 131, 318, 121, 88,  99,  327, 350, 305, 371, 393, 206, 262, 73,  332, 66,  367, 110, 267, 105, 381, 83,
                   52,  314, 302, 255, 79,  129, 101, 80,  284, 192, 253, 221, 64,  299, 144, 138, 183, 234, 291, 314, 261, 72,  183, 190, 269, 163, 90,  209, 166, 158, 285, 167, 103, 375,
                   260, 297, 225, 389, 89,  132, 79,  178, 223, 261, 136, 356, 120, 192, 199, 204, 88,  392, 271, 152, 276, 385, 141, 156, 79,  282, 141, 156, 79,  141, 156, 79},
            .rhoR = {2, 4, 4, 2, 2, 2, 5, 3, 5, 1, 2, 5, 4, 3, 5, 3, 2, 1, 5, 5, 5, 3, 5, 1, 4, 1, 1, 3, 2, 3, 5, 1, 2, 4, 3, 4, 1, 3, 5, 5, 5, 4, 4, 4, 1, 2, 2, 2, 3, 3,
                     2, 1, 4, 5, 5, 5, 2, 2, 5, 2, 5, 2, 2, 1, 2, 4, 5, 5, 5, 1, 5, 4, 4, 3, 4, 3, 4, 4, 5, 5, 3, 2, 5, 4, 5, 2, 3, 3, 3, 4, 5, 1, 3, 3, 2, 4, 6, 3, 7, 4},
            .pR = {93644,  39649,  339174, 112878, 259384, 343164, 233259, 317285, 127482, 197816, 234582, 177675, 326625, 28916,  346530, 322863, 395569, 14075,  388938, 206910,
                   387186, 282753, 232149, 21634,  229816, 84835,  40524,  122346, 362071, 124405, 169407, 179133, 109780, 312619, 142633, 238097, 172750, 105307, 53896,  145872,
                   188215, 375194, 91366,  357592, 8055,   22759,  288328, 318824, 239411, 209427, 219542, 201244, 172415, 344459, 45137,  261801, 339786, 150532, 293557, 239951,
                   180570, 310078, 5490,   288644, 44127,  138407, 355253, 163598, 40291,  320368, 17585,  101370, 298964, 182106, 46839,  23049,  90916,  99261,  75995,  235761,
                   48242,  166361, 142876, 356127, 60717,  97634,  113278, 286284, 48080,  93500,  238665, 188704, 62657,  44030,  48080,  93500,  238665, 188704, 62657,  44030},
            .expectedMassFlux = {0,           196.580904,  -156.605344, -91.4071,    -21.41607,   -319.65192,  -7.42896,     236.174576,   -70.422,     -18.011126,  1918.492158, -3087.8936,
                                 -48.69,      -507.423915, 40.72194,    619.0224,    0,           3633.40932,  950.58032,    -1713.138,    0,           142.21755,   65.270286,   -384.062074,
                                 -2034.96448, 52.73072,    -332.585316, -309.28491,  81.256815,   -137.37108,  -2298.78723,  -142.95729,   -151.057356, -382.359752, -141.041628, -7.411656,
                                 1592.39613,  -273.160845, 153.943024,  -1229.02299, 2162.7165,   1594.467072, -1417.549648, -1180.876032, 106.7028,    799.350832,  1114.77129,  10.399664,
                                 369.64004,   -500.981814, -701.676534, -224.24454,  1123.158816, 60.4464,     50.4872,      247.62702,    -61.703574,  -424.68838,  1237.20168,  68.469464,
                                 -98.05635,   -313.539292, 227.69576,   171.359496,  -33.69612,   -395.78666,  51.137649,    -1220.79375,  -677.8135,   78.80448,    265.43286,   -1224.838076,
                                 0,           28.6744,     2783.16144,  372.519868,  -324.046652, -1034.17596, -1163.87576,  1353.671514,  -275.84676,  311.356064,  1585.2749,   -105.423936,
                                 -455.6772,   984.56928,   66.61869,    134.023645,  -191.33424,  -305.56988,  898.097772,   -183.604668,  1033.7973,   1217.727624, 195.606537,  -386.835696,
                                 1872.87585,  -226.494504, 246.45075,   -188.744588},
            .expectedInterfacePressure = {0,           275817.883,  593902,      54900.36373, 225210.7427, 392808.8396, 437166.4231, 511002.4354, 20354.03108, 36674.09732, 212864.9512, 282157.7441,
                                          434845,      28916,       214680.4982, 201212.6997, 0,           221364,      300401,      206840.8921, 0,           216116.0923, 508802,      86052.19808,
                                          229816,      424.832859,  40524,       122346,      124084.1285, 141494.0278, 169407,      179133,      45908.56886, 398946.3602, 147132.3236, 3999.315309,
                                          202981.982,  162584.6514, 201154.2742, 145872,      492607.0086, 520864.6152, 91366,       380096.156,  204077.1747, 267472.7058, 643768,      77988.97212,
                                          184743,      562660.4259, 230704.4333, 492744.3105, 304699.8756, 419995,      71717.82725, 270855.5112, 124811.2129, 150532,      299086,      50505.45696,
                                          81306.15675, 486004.3742, 210283.9492, 557631.2384, 39438.49987, 152292.4205, 120922.6938, 387636,      54751.27618, 11536.97843, 113917.1656, 97793.15955,
                                          0,           360662.0678, 133120,      72757.37657, 94535.48702, 99254.34951, 165560.4233, 289661,      396723,      19176.42771, 154440,      104955.2566,
                                          128601.8358, 218854,      64521.62124, 38022.58299, 23124.90334, 333485.8332, 266589.825,  188704,      108697.9922, 66058.27781, 99436,       484091.0114,
                                          122771.9279, 253915.663,  180269.541,  66641.95381},
            .expectedDirection = {LEFT,  LEFT,  RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, LEFT,  RIGHT, RIGHT, LEFT,  RIGHT, RIGHT, RIGHT, LEFT,  LEFT,  LEFT,  LEFT,  LEFT,  RIGHT,
                                  LEFT,  LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  RIGHT, RIGHT, LEFT,  RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, LEFT,  RIGHT, LEFT,  RIGHT,
                                  LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  LEFT,  LEFT,  LEFT,  LEFT,  RIGHT, RIGHT, RIGHT, LEFT,  LEFT,  LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  LEFT,
                                  RIGHT, RIGHT, LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  RIGHT, RIGHT, LEFT,  LEFT,  RIGHT, LEFT,  LEFT,  LEFT,  LEFT,  RIGHT, RIGHT, RIGHT, LEFT,
                                  RIGHT, LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  LEFT,  LEFT,  RIGHT, RIGHT, LEFT,  RIGHT, LEFT,  LEFT,  LEFT,  RIGHT, LEFT,  RIGHT, LEFT,  RIGHT}

        },
        (FluxCalculatorTestParameters){.testName = "AverageFlux",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AverageFlux>(),
                                       .uL = {80, 168.5, -161.4, 76},
                                       .aL = {80, 337, 269, 190},
                                       .rhoL = {2, 4, 4, 2},
                                       .pL = {93644, 39649, 339174, 112878},
                                       .uR = {32, 31.2, -68, -122},
                                       .aR = {32, 52, 136, 305},
                                       .rhoR = {1, 3, 5, 6},
                                       .pR = {51426, 264102, 254728, 195151},
                                       .expectedMassFlux = {96, 383.8, -492.8, -290},
                                       .expectedInterfacePressure = {72535, 151875.5, 296951, 154014.5},
                                       .expectedDirection = {NA, NA, NA, NA}

        },
        // Rieman flux testing
        (FluxCalculatorTestParameters){
            .testName = "RiemanFlux",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Rieman>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.18321596, 0.74833148, 37.4165739, 0.1183216, 10.3708995},  // gam=1.4 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.05830052, 0.74833148, 0.1183216, 11.8321596, 3.28163145},  // gam=1.4 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.39539107, 0, 11.2697554, -3.56358518796, 117.5701},           // status at x =0
            .expectedInterfacePressure = {0.30313018, 0.00189387, 460.893787, 46.095, 460.894},  // pressure at x=0
            .expectedDirection = {LEFT, RIGHT, LEFT, RIGHT, LEFT}                                // Upwind direction based on velocity at x = 0
        },
        // Riemann2Gas flux testing, same gamma L/R
        (FluxCalculatorTestParameters){
            .testName = "Riemann2GasFlux",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Riemann2Gas>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.18321596, 0.74833148, 37.4165739, 0.1183216, 10.3708995},  // gam=1.4 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.05830052, 0.74833148, 0.1183216, 11.8321596, 3.28163145},  // gam=1.4 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.39539107, 0, 11.2697554, -3.56358518796, 117.5701},           // status at x =0
            .expectedInterfacePressure = {0.30313018, 0.00189387, 460.893787, 46.095, 460.894},  // pressure at x=0
            .expectedDirection = {LEFT, RIGHT, LEFT, RIGHT, LEFT}                                // Upwind direction based on velocity at x = 0
        },
        // Riemann2Gas flux testing, gamma 1.4 L/ gamma 1.667 R
        (FluxCalculatorTestParameters){
            .testName = "Riemann2GasFluxT2",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Riemann2Gas>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.667"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.18321596, 0.74833148, 37.4165739, 0.1183216, 10.3708995},  // gam=1.4 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.15481600, 0.81657823, 0.12911235, 12.91123542, 3.58091149},  // gam=1.667 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.39442313, 0.00001119, 11.0900218667, -3.61787265, 117.570105900},    // status at x =0
            .expectedInterfacePressure = {0.314396658, 0.000506098, 475.022995502, 43.1357, 460.8940},  // pressure at x=0
            .expectedDirection = {LEFT, LEFT, LEFT, RIGHT, LEFT}                                        // Upwind direction based on velocity at x = 0
        },
        // Riemann2Gas flux testing, gamma 1.667 L/ gamma 1.4 R
        (FluxCalculatorTestParameters){
            .testName = "Riemann2GasFluxT3",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::Riemann2Gas>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.667"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.2911235, 0.816578, 40.828911, 0.12911235, 11.3167106},  // gam=1.667 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.05830052, 0.74833148, 0.1183216, 11.8321596, 3.28163145},  // gam=1.4 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.405243, -0.005215, 11.4486735, -3.5061155, 112.3130},                // status at x =0
            .expectedInterfacePressure = {0.28296141, 0.000676, 430.992964, 47.5230682, 1758.5562536},  // pressure at x=0
            .expectedDirection = {LEFT, RIGHT, LEFT, RIGHT, LEFT}                                       // Upwind direction based on velocity at x = 0
        },
        // Riemann2Gas flux testing, same gamma L/R
        (FluxCalculatorTestParameters){
            .testName = "RiemannStiffFlux",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.18321596, 0.74833148, 37.4165739, 0.1183216, 10.3708995},  // gam=1.4 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.05830052, 0.74833148, 0.1183216, 11.8321596, 3.28163145},  // gam=1.4 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.39539107, 0, 11.2697554, -3.56358518796, 117.5701},           // status at x =0
            .expectedInterfacePressure = {0.30313018, 0.00189387, 460.893787, 46.095, 460.894},  // pressure at x=0
            .expectedDirection = {LEFT, RIGHT, LEFT, RIGHT, LEFT}                                // Upwind direction based on velocity at x = 0
        },
        // RiemannStiff flux testing, gamma 1.4 L/ gamma 1.667 R
        (FluxCalculatorTestParameters){
            .testName = "RiemannStiffFluxT2",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.667"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.18321596, 0.74833148, 37.4165739, 0.1183216, 10.3708995},  // gam=1.4 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.15481600, 0.81657823, 0.12911235, 12.91123542, 3.58091149},  // gam=1.667 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.39442313, 0.00001119, 11.0900218667, -3.61787265, 117.570105900},    // status at x =0
            .expectedInterfacePressure = {0.314396658, 0.000506098, 475.022995502, 43.1357, 460.8940},  // pressure at x=0
            .expectedDirection = {LEFT, LEFT, LEFT, RIGHT, LEFT}                                        // Upwind direction based on velocity at x = 0
        },
        // Riemann2Gas flux testing, gamma 1.667 L/ gamma 1.4 R
        (FluxCalculatorTestParameters){
            .testName = "RiemannStiffFluxT3",
            .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.667"}})),
                std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
            .uL = {0.0, -2.0, 0.0, 0.0, 19.5975},
            .aL = {1.2911235, 0.816578, 40.828911, 0.12911235, 11.3167106},  // gam=1.667 a = \gam * p / \rho
            .rhoL = {1.0, 1.0, 1.0, 1.0, 5.99924},
            .pL = {1.0, 0.4, 1000.0, 0.01, 460.894},
            .uR = {0.0, 2.0, 0.0, 0.0, -6.19633},
            .aR = {1.05830052, 0.74833148, 0.1183216, 11.8321596, 3.28163145},  // gam=1.4 a = \gam * p / \rho
            .rhoR = {0.125, 1.0, 1.0, 1.0, 5.99242},
            .pR = {0.1, 0.4, 0.01, 100.0, 46.0950},
            .expectedMassFlux = {0.405243, -0.005215, 11.4486735, -3.5061155, 112.3130},                // status at x =0
            .expectedInterfacePressure = {0.28296141, 0.000676, 430.992964, 47.5230682, 1758.5562536},  // pressure at x=0
            .expectedDirection = {LEFT, RIGHT, LEFT, RIGHT, LEFT}                                       // Upwind direction based on velocity at x = 0
        },
        // RiemannStiff flux testing, water/air
        (FluxCalculatorTestParameters){.testName = "RiemannStiffFluxT4",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(
                                           std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                               {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645e9"}})),
                                           std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}}))),
                                       .uL = {0.0},
                                       .aL = {2044.9484101072085},
                                       .rhoL = {1000.0},
                                       .pL = {1000000000.0},
                                       .uR = {0.0},
                                       .aR = {52.91502622129181},
                                       .rhoR = {50},
                                       .pR = {100000.0},
                                       .expectedMassFlux = {434205.19124982634},
                                       .expectedInterfacePressure = {21306365.4962665},
                                       .expectedDirection = {LEFT}},
        // RiemannStiff flux testing, air/water
        (FluxCalculatorTestParameters){.testName = "RiemannStiffFluxT5",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(
                                           std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma", "1.4"}})),
                                           std::make_shared<ablate::eos::StiffenedGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{
                                               {"gamma", "1.932"}, {"Cp", "8095.08"}, {"p0", "1.1645e9"}}))),
                                       .uL = {0.0},
                                       .aL = {52.91502622129181},
                                       .rhoL = {50},
                                       .pL = {100000.0},
                                       .uR = {0.0},
                                       .aR = {2044.9484101072085},
                                       .rhoR = {1000},
                                       .pR = {1000000000.0},
                                       .expectedMassFlux = {-434205.19124982634},
                                       .expectedInterfacePressure = {21306365.4962665},
                                       .expectedDirection = {RIGHT}},
        (FluxCalculatorTestParameters){.testName = "OffFlux",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::OffFlux>(),
                                       .uL = {80, 168.5, -161.4, 76},
                                       .aL = {80, 337, 269, 190},
                                       .rhoL = {2, 4, 4, 2},
                                       .pL = {93644, 39649, 339174, 112878},
                                       .uR = {32, 31.2, -68, -122},
                                       .aR = {32, 52, 136, 305},
                                       .rhoR = {1, 3, 5, 6},
                                       .pR = {51426, 264102, 254728, 195151},
                                       .expectedMassFlux = {0.0, 0.0, 0.0, 0.0},
                                       .expectedInterfacePressure = {0.0, 0.0, 0.0, 0.},
                                       .expectedDirection = {NA, NA, NA, NA}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpHighSpeedLeftToRight",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(0),
                                       .uL = {00.0, 500.0, 1500.0, 2500.0, 3500.0, 4500.0},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.498047080891},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {500.0, 1500.0, 2500.0, 3500.0, 4500.0, 5000.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {280.0321496650917, 1013.8750522048633, 3043.574457950678, 5075.851480605753, 7110.680674790922, 9148.036928946367},
                                       .expectedInterfacePressure = {125809.91038349853, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .expectedDirection = {LEFT, LEFT, LEFT, LEFT, LEFT, LEFT}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpHighSpeedRightToLeft",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(0),
                                       .uL = {0.0, -500.0, -1500.0, -2500.0, -3500.0, -4500.0},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {-500.0, -1500.0, -2500.0, -3500.0, -4500.0, -5000.0, -5000.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {-695.4374555207759, -3043.574457950678, -5075.851480605753, -7110.680674790923, -9148.036928946367, -10167.654946084249},
                                       .expectedInterfacePressure = {527669.7617208518, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedDirection = {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpLowSpeedLeftToRight",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(0),
                                       .uL = {0.0, 0.5, 1.5, 2.5, 3.5, 4.5},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.498047080891},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {0.5, 1.5, 2.5, 3.5, 4.5, 5.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {0.16669052478808, 1.4140255203600878, 3.4452931723016658, 5.479131476380215, 7.515514952189054, 9.351066525194593},
                                       .expectedInterfacePressure = {251513.141336571, 251881.783102883, 252877.7587264395, 253873.76531142398, 254869.8028001732, 255993.15745896765},
                                       .expectedDirection = {LEFT, LEFT, LEFT, LEFT, LEFT, LEFT}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpLowSpeedRightToLeft",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(0),
                                       .uL = {0.0, -0.5, -1.5, -2.5, -3.5, -4.5},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {-0.5, -1.5, -2.5, -3.5, -4.5, -5.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {-0.8243234947604119, -2.6431675433887167, -4.6738771785086595, -6.707145948584977, -8.742948777070168, -9.964562041554487},
                                       .expectedInterfacePressure = {252226.83514314616, 253359.28463113605, 254363.30986685975, 255367.30410115453, 256371.26739198447, 256746.84137967348},
                                       .expectedDirection = {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpLowSpeedLeftToRightWithPgs",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(
                                           0, std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 100, NAN)),
                                       .uL = {0.0, 0.5, 1.5, 2.5, 3.5, 4.5},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.498047080891},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {0.5, 1.5, 2.5, 3.5, 4.5, 5.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {0.44282254247899977, 2.0029606409670233, 3.925231119301933, 5.6722932454341395, 7.223867983781705, 9.146014361217187},
                                       .expectedInterfacePressure = {21.811041262059153, 19.45032886696595, 21.70840295490869, 24.119786383120882, 25.426157631414622, 25.611982076699707},
                                       .expectedDirection = {LEFT, LEFT, LEFT, LEFT, LEFT, LEFT}

        },
        (FluxCalculatorTestParameters){.testName = "AusmpUpLowSpeedRightToLeftWithPgs",
                                       .fluxCalculator = std::make_shared<ablate::finiteVolume::fluxCalculator::AusmpUp>(
                                           0, std::make_shared<ablate::finiteVolume::processes::PressureGradientScaling>(std::shared_ptr<ablate::eos::EOS>{}, 100, NAN)),
                                       .uL = {0.0, -0.5, -1.5, -2.5, -3.5, -4.5},
                                       .aL = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoL = {1.783191515808363, 2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415},
                                       .pL = {251619.82076699706, 252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706},
                                       .uR = {-0.5, -1.5, -2.5, -3.5, -4.5, -5.0},
                                       .aR = {419.1161009333595, 407.21760127816424, 407.8271236675531, 408.43558618161376, 409.04299399830086, 409.4980470808913},
                                       .rhoR = {2.0277501044097264, 2.0290496386337855, 2.0303405922423012, 2.0316230499402637, 2.032897095321415, 2.0335309892168496},
                                       .pR = {252119.82076699706, 253119.82076699706, 254119.82076699706, 255119.82076699706, 256119.82076699706, 256619.82076699706},
                                       .expectedMassFlux = {-0.5103209197483072, -2.0538549890223545, -4.1936338779020685, -6.5138621695596655, -9.034778639382575, -10.169678144486442},
                                       .expectedInterfacePressure = {28.89470449374076, 32.32135045739733, 29.83487717596696, 27.115889289695332, 25.718780282784518, 25.661982076699704},
                                       .expectedDirection = {RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT}

        }),
    [](const testing::TestParamInfo<FluxCalculatorTestParameters>& info) { return info.param.testName; });
