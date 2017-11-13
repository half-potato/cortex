import numpy as np
import math

THRESHOLD = -0.1
RESTING_POTENTIAL = -0.7
FIRE_POTENTIAL = 0.4
# REPOLARIZATION
REPO_RATE = -0.4
#REPO_OVERSHOOT = -0.9
#if discreet
REPO_OVERSHOOT = math.ceil(RESTING_POTENTIAL / REPO_RATE) * REPO_RATE
REPO_PERIOD = (REPO_OVERSHOOT-FIRE_POTENTIAL) / REPO_RATE
print("Repolarization period: %i" % REPO_PERIOD)
# REFRACTORY
REFRACT_RATE = 0.1
REFRACT_PERIOD = math.ceil((RESTING_POTENTIAL - REPO_OVERSHOOT) / REFRACT_RATE)
print("Refractory period: %i" % REFRACT_PERIOD)

# This is more of a model. Not to be used in the actual code
class Neuron:
    def __init__(self):
        self.lastFiring = 9999
        self.kill = False
        self.potential = RESTING_POTENTIAL

    def update(self, change, dt=1):
        if self.lastFiring >= 0:
            self.lastFiring += dt
        if self.lastFiring > REPO_PERIOD+REFRACT_PERIOD+dt:
            self.potential += change
        # If brought above threshold after the refractory period, fire
        if self.potential > THRESHOLD and \
           self.lastFiring > REPO_PERIOD+REFRACT_PERIOD+dt:
            self.lastFiring = 0.0
            self.potential = FIRE_POTENTIAL
            return 1
        # if during repolarizing period, bring potential to REPO_OVERSHOOT
        elif self.potential > REPO_OVERSHOOT and \
             self.lastFiring < REPO_PERIOD:
            self.potential += REPO_RATE*dt
        # if not repolarizing, bring potential closer to resting
        diff = float(np.squeeze(self.potential - RESTING_POTENTIAL))
        if abs(diff) > REFRACT_RATE/2 and \
             self.lastFiring > REPO_PERIOD:
            rate = min(REFRACT_RATE, diff)
            sign = (diff) / abs(diff)
            self.potential -= sign * abs(rate) * dt
        return 0
