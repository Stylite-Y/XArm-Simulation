from mujoco_py import load_model_from_path, MjSim, MjViewer
import os
import math

xml_path = os.path.dirname(os.path.realpath(__file__)) + '/raisim_examp/heightMapUsingPng.xml'
model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)
t = 0
while True:
    sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
    sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
