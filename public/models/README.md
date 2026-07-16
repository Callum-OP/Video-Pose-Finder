# Character model

The rigged humanoid shown in the 3D Pose Editor is the bundled
**Low Poly Male Base - Slender** by [Mesh-Base](https://sketchfab.com/mesh-base)
([Sketchfab](https://sketchfab.com/3d-models/low-poly-male-base-slender-9a6fd72aa31540f2a6a8e8d236778a2f),
CC-BY-4.0). It lives at `public/low_poly_male_base_-_slender.glb` — *not* in this
folder — because this folder is git-ignored and the model must ship with the
GitHub Pages deploy.

The editor's source of truth is the captured pose landmarks, not this rig — the
character is a *display-only skin* driven by `src/utils/retargetRig.js`.

## Using your own character

The editor loads, in order: **`character.fbx`**, `character.glb`, then the bundled
example. So to use your own model, drop a rigged file here as `character.fbx` (or
`character.glb`) and it's picked up automatically — no code change needed. This
folder is git-ignored, so your model stays local.

- Mixamo rigs work: bone names are matched by suffix, so `mixamorig:LeftArm`,
  `mixamorigLeftArm`, `Armature|mixamorig:LeftArm`, or bare `LeftArm` all resolve.
- UE/Blender-style rigs (`pelvis`, `spine01`, `upperarm_L`, `thigh_r`, ...) work
  too — they're aliased onto the Mixamo names in `retargetRig.js`.
- The rig is auto-scaled to the captured figure, so file units don't matter (glTF is
  usually metres, Mixamo FBX usually centimetres).
- Rigs with unrecognised bone names fall back to the capsule mannequin.
