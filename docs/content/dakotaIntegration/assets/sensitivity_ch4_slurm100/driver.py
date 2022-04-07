import dakota.interfacing as di
import numpy as np
import os
import subprocess


#-----------------------------------------------------------------
params, results = di.read_parameters_file()
x1=params['x1']*1E+15
x2=params['x2']*1E+9

#-----------------------------------------------------------------

with open ('ORIGINAL_2S_CH4_CM2.mech.dat', "r") as myfile:
    inputfile = myfile.readlines()
    #print(inputfile[19])
    inputfile[19] = 'CH4+1.5O2=>CO+2H2O '+str(x1)+'  0.00   35000.00\n'
    inputfile[22] = 'CO+0.5O2<=>CO2 '+str(x2)+'  0.000   12000.00\n'
    #for i in range(len(inputfile)):
    #        inputfile[i] = inputfile[i][0:-1]
    np.savetxt('2S_CH4_CM2.mech.dat', inputfile, fmt='%s',delimiter='')

#-----------------------------------------------------------------
command = '$ABLATE_DIR/ablate --input ignitionDelay2S_CH4_CM2.yaml  '

p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)

(stdout, err) = p.communicate()

output = stdout.splitlines()


#-----------------------------------------------------------------
with open ('_ignitionDelay2S_CH4_CM2/ignitionDelayTemperature.txt', "r") as myoutfile:
    outfile = myoutfile.readlines()
    targetline = outfile[0].split(':')
    QoI = float(targetline[1])
    print(QoI)

os.remove('2S_CH4_CM2.mech.dat')
os.remove('_ignitionDelay2S_CH4_CM2/ignitionDelayTemperature.txt')


#-----------------------------------------------------------------
for i, r in enumerate(results.responses()):
    r.function = QoI
results.write()
