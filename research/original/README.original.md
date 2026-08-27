# Product Design System with Generative AI and 3D Voxel Conversion

This project uses Generative AI and 3D modeling techniques to design customized products based on user input. The system transforms text prompts into 2D images and converts them into voxel-based 3D representations, making them compatible with CAD applications.

---

## **Features**
- **Generative AI for Image Creation**: Generates 2D images based on user-defined product requirements using models like Stable Diffusion.
- **Text-to-Design Understanding**: NLP models (e.g., BERT, RoBERTa) process user prompts to guide design creation.
- **2D to 3D Conversion**: Converts generated images into voxel grids for further 3D modeling.
- **Integration with CAD Tools**: Enables seamless refinement and optimization of designs for manufacturing or prototyping.

---

## **How It Works**

### **Step 1: Image Generation**
- **Goal**: Generate a visual representation of the user's product design.  
- **Process**:
  1. User provides a prompt, e.g., "Design a car for a person with one leg."
  2. NLP models process the input to guide image generation.
  3. Generative AI (e.g., Stable Diffusion) produces a 2D image based on the prompt.

---

### **Step 2: Convert Image into Voxel Data**
- **Goal**: Transform the generated 2D image into a 3D voxel representation.  
- **Process**:
  1. Use tools like **Open3D**, **MeshLab**, or **Matplotlib 3D** for voxel generation.
  2. Map the 2D image onto a voxel grid, creating a pixelated 3D design.
  3. Smooth or optimize voxel data to ensure compatibility with CAD applications.

---

## **Technologies Used**
- **Generative AI**: Stable Diffusion, PyTorch
- **Natural Language Processing**: BERT, RoBERTa
- **3D Modeling and Visualization**: Open3D, PyTorch3D, Trimesh, Matplotlib 3D

---

## **Getting Started**

### **Prerequisites**
- Python 3.8+
- Required Python libraries: `torch`, `transformers`, `diffusers`, `open3d`, `trimesh`, and `matplotlib`

### **Installation**
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/your-repo-name.git


