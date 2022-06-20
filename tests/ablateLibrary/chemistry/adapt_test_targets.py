import numpy as np
import pandas as pd
from tensorflow import keras
import os

model_path = f'../inputs/chemistry/chemTabTestModel_1' 
#'/project/chandola/ablate/tests/ablateLibrary/inputs/chemistry/chemTabTestModel_1/'

W = pd.read_csv(f'{model_path}/weights.csv', index_col=0)
Winv = pd.read_csv(f'{model_path}/weights_inv.csv', index_col=0)
print_array = lambda x: ','.join([str(i) for i in np.asarray(x).squeeze()]) 
print_str_array = lambda x: ','.join([f'"{i}"' for i in np.asarray(x).squeeze()]) 
print(W)

input_mass = m = np.linspace(0.1,5.3,53)
m = m[:,np.newaxis]
output_cpv = cpv = np.dot(W.T,m).flatten()
print('CPVs <-- mass_fractions:')
print('CPVs:', print_array(cpv), '<-- mass_fractions:', print_array(m))

# use only 1 input CPV for both tests (for simplicity)
input_cpv = cpv = np.linspace(0.1,1,len(cpv)) 
cpv = cpv[:,np.newaxis]
output_mass = m = np.dot(Winv.T,cpv).flatten()
print('CPVs --> mass_fractions:')
print('CPVs:', print_array(cpv), '--> mass_fractions:', print_array(m))

regressor = keras.models.load_model(f"{model_path}/regressor")
source_output = regressor(cpv.T)
souener_output = source_output['static_source_prediction'].numpy()[0,0]
# Source energy is always first static source term!

print('CPVs --> Source Terms:')
print(f'CPVs: {print_array(cpv)} -->')
print('pred CPV source terms: ', print_array(source_output['dynamic_source_prediction']))
print('pred Source Energy: ', souener_output) 

macro_header=f"""
#define CPV_NAMES "zmix", {print_str_array(W.columns)} 
#define SPECIES_NAMES {print_str_array(W.index)} 

#define INPUT_CPVS {print_array(input_cpv)} 
#define OUTPUT_MASS_FRACTIONS {print_array(output_mass)} 

#define INPUT_MASS_FRACTIONS {print_array(input_mass)} 
#define OUTPUT_CPVS {print_array(output_cpv)} 

#define OUTPUT_SOURCE_TERMS {print_array(source_output['dynamic_source_prediction'])} 
#define OUTPUT_SOURCE_ENERGY {souener_output} 
"""

print('\n'+'='*50)
print('macro header:')
print('='*50)
print(macro_header)

with open('test_targets.h', 'w') as f:
    f.write(macro_header)
