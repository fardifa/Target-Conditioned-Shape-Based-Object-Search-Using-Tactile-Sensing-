import time
import mujoco
import mujoco.viewer

xml = """
<mujoco>
  <worldbody>
    <geom type="plane" size="2 2 .1" rgba="0.9 0.95 0.9 1"/>
    <body name="cube" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" rgba="0.1 0.5 0.9 1"/>
      <joint type="free"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # runs until you press ESC or close the window
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model, data)  # advance physics one step
        viewer.sync()                 # render a frame

        # keep sim time ~ real time
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
