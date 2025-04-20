
import bpy
import sys

# Get the user input from command-line arguments
user_input = sys.argv[-1]

# Delete default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Generate a 3D model based on user input
if "chair" in user_input.lower():
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    bpy.ops.transform.resize(value=(1, 1, 0.5))
elif "table" in user_input.lower():
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
    bpy.ops.transform.resize(value=(2, 2, 0.2))
else:
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))

# Export the model as an OBJ file
output_path = "/content/generated_model.obj"
bpy.ops.export_scene.obj(filepath=output_path)

print(f"3D model exported to {output_path}")