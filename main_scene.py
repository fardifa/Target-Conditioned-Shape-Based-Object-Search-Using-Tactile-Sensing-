from pathlib import Path
import mujoco
import mujoco.viewer
import os

def main():
    project_root = Path(__file__).parent.resolve()
    #scene_path = project_root / "panda_scene.xml"
    scene_path = project_root / "shadow_hand_scene.xml"

    print(f"Loading scene: {scene_path}")

    if not scene_path.exists():
        print("Scene file not found!")
        return

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
        return

    print("Launching MuJoCo viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)

if __name__ == "__main__":
    main()
