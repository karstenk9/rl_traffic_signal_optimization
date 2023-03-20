import os
import sys

sys.path.append(os.path.join('c:', os.sep, '/opt/homebrew/opt/sumo/share/sumo/tools'))

import traci
print("LOADPATH:", '\n'.join(sys.path))
print("TRACIPATH:", traci.__file__)
sys.exit()

sumoBinary = 'Applications/sumo-gui'
traci.start([sumoBinary,'-c','run.sumocfg','--save-configuration','debug.sumocfg'])
