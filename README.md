# Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing-
This repository contains the project code for ECE 699: Robot Learning.  
The project explores tactile-based object search and recognition using reinforcement learning and simulation in MuJoCo.

---

## Environment Setup **Note:** This setup is tested on macOS (Apple Silicon), specifically a MacBook Pro M3.


### 1. Create a Conda Virtual Environment
```bash
conda create -n rlsim python=3.11 -y
conda activate rlsim
```

### 2. Install MuJoCo and other dependencies

Install the required system and Python dependencies:

```bash
brew install cmake glfw

pip install mujoco==3.1.6 mujoco-python-viewer gymnasium numpy pillow imageio matplotlib \
torch torchvision torchaudio 'stable-baselines3[extra]==2.3.2' pandas scikit-learn tqdm tensorboard pyyaml seaborn
```
### 3. Install the MuJoCo Desktop App (optional but recommended)

Download the .dmg installer for macOS from
🔗 https://github.com/google-deepmind/mujoco/releases

Open the .dmg and drag MuJoCo.app into your Applications folder.

### 4. Clone and Organize the Project Repository
Clone this repository and navigate into it:
```
git clone https://github.com/<your-username>/Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing.git
cd Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing
```

<pre> ```
project/
│
├── main_scene.py                 # Python launcher for MuJoCo scenes
├── panda_scene.xml               # Scene with Franka Panda arm + table + objects
├── shadow_hand_scene.xml         # Scene with Shadow Hand setup for tactile probing
│
├── mujoco_menagerie/             # Prebuilt robot models from DeepMind’s MuJoCo Menagerie
│   │
│   ├── franka_emika_panda/
│   │   ├── panda.xml             # Panda arm model definition
│   │   └── assets/               # Meshes and textures for the Panda robot
│   │       ├── panda_link0.obj
│   │       ├── panda_link1.obj
│   │       └── ...
│   │
│   └── shadow_hand/
│       ├── left_hand.xml         # Shadow Hand model definition
│       └── assets/               # Meshes and textures for the Shadow Hand
│           ├── forearm_0.obj
│           ├── palm.obj
│           ├── f_proximal.obj
│           └── ...
│
└── assets/                       # (Optional) additional custom assets for your own scenes
    ├── textures/
    └── meshes/
``` </pre>

### 5. Modify the MuJoCo Files 
Before running, ensure you’ve made the required XML modifications listed below.
The default MuJoCo Menagerie files from Google need small edits for compatibility and correct positioning in this project.

File:
mujoco_menagerie/franka_emika_panda/panda.xml

Change the <compiler> paths to absolute paths on your system (so MuJoCo can locate all .obj meshes):

```
<mujoco model="panda">
  <compiler 
      angle="radian" 
      meshdir="/Users/<your-username>/.../ECE 699 Robot Learning/project/mujoco_menagerie/franka_emika_panda/assets"
      autolimits="true"/>
```
File:
mujoco_menagerie/shadow_hand/left_hand.xml

Make these minimal but necessary edits:

I. Set absolute mesh path
```
<compiler 
    angle="radian"
    meshdir="/Users/<your-username>/.../ECE 699 Robot Learning/project/mujoco_menagerie/shadow_hand/assets"
    autolimits="true"/>
```
II. Adjust root body position and orientation so the hand rests just above the table:
Find the line:
```
<body name="lh_forearm" childclass="left_hand" quat="0 1 0 1">
```
Replace it with:
```
<body name="lh_forearm"
      childclass="left_hand"
      pos="0 0 0.72"
      quat="0 0.707 0 0.707">
```

### 6. Modify the MuJoCo Files 

Once the environments and dependencies are set up, you can visualize the robot scenes directly in the MuJoCo viewer.

```
mjpython main_scene.py 
```
Change the scene path to either "panda_scene.xml" or "shadow_hand.xml" as per requirement.














