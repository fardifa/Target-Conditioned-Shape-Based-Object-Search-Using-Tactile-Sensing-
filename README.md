# Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing-
This repository contains the project code for ECE 699: Robot Learning.  
The project explores tactile-based object search and recognition using reinforcement learning and simulation in MuJoCo.

---

## Environment Setup (For macOS (Apple Silicon))


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
├── dataset/ # Scripts + data used to generate tactile images and train classifier
│ ├── create_tactile_data.py
│ ├── train_classifier.py
│ └── best_tactile_classifier.pth
│
├── mujoco_menagerie/ # (Optional) Menagerie robot models used during early experiments
│ └── ... # Not required for final tactile search system
│
├── main_scene.py # Early MuJoCo scene launcher (not used in final pipeline)
├── panda_scene.xml # Early Panda-arm scene (archived)
├── shadow_hand_scene.xml # Early Shadow Hand scene (archived)
│
└── final_scene/ # Final active tactile search system
    │
    ├── main.py                       # Manual tactile search (fixed policy)
    ├── multi_object_search_ppo.py    # PPO-based multi-object tactile search
    │
    ├── Search_scene.xml              # Final MuJoCo tactile scene (finger + 4 objects)
    │
    ├── best_tactile_classifier_convnet.pth   # Trained tactile classifier
    ├── 1000_ppo_policy.pth                   # Example trained PPO policy
    │
    ├── motion_controller.py          # Tactile probing controller (touch cycles, pose control)
    ├── tactile_env.py                # PPO environment wrapper for tactile search
    ├── search_manager.py             # Bayesian incremental fusion of classifier predictions
    ├── classifier.py                 # Classifier interface for ConvNeXt model
    │
    └── (other helper modules)        # utils, tactile_image_buffer, etc.

``` </pre>

### 5. Run the Final Tactile Search System (inside `final_scene/`)

All final runnable scripts for this project are located in the `final_scene/` directory.  
Navigate into the folder:

```bash
cd final_scene
```
#### 5.1 Run Baseline Manual Active Search

To run the manual baseline version, run:

```bash
mjpython main.py
```

When prompted:

```bash
Select mode (manual/ppo/train_ppo): manual
Enter target object: cube
```

This will:

* Load the MuJoCo tactile scene (Search_scene.xml)

* Touch each object (sphere → cube → cylinder → cone)

* Generate pseudo-GelSight images during contact

* Fuse classifier predictions incrementally

* Stop early if the target is confidently identified

#### 5.2 Multi-Object PPO Tactile Search (Reinforcement Learning)
This mode runs the full target-conditioned tactile search using a trained PPO policy.

Run:

```bash
mjpython multi_object_search_ppo.py
```

When prompted:

```bash
Enter target object for PPO multi-object search: cube
Enter PPO model path: 1000_ppo_policy.pth
```
This will:

* Initialize PPO environment (tactile_env.py)

* Run multi-step tactile episodes

* Learn when to explore, reject, or confirm

