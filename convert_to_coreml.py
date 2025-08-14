import torch
import cv2
import numpy as np
import qai_hub as hub
from qai_hub_models.models.facemap_3dmm import Model
from PIL import Image

IMAGE_PATH = "assets/face_img.jpg"
# 1. Load the pretrained model
torch_model = Model.from_pretrained()
torch_model.eval()

# 2. Get model's expected input size (should be 128x128 for this one)
input_shape = torch_model.get_input_spec()
print("Expected input spec:", input_shape)

# 3. Load an image

img = cv2.imread(IMAGE_PATH)  # BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# 4. Resize to match model input
img_resized = cv2.resize(img, (128, 128))

# 5. Convert to float tensor in (C, H, W) format
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()

# 6. Normalize if the model expects it (common: [0, 1] range)
img_tensor = img_tensor / 255.0  

# 7. Add batch dimension
img_tensor = img_tensor.unsqueeze(0)  # Shape: (1, 3, 128, 128)

# 8. Run inference
with torch.no_grad():
    output = np.array(torch_model(img_tensor)).flatten()

print("Output shape:", output.shape)

alpha_id, alpha_exp, pitch, yaw, roll, tX, tY, f = (
        output[0:219],
        output[219:258],
        output[258],
        output[259],
        output[260],
        output[261],
        output[262],
        output[263],
)

print("roll for pt: ", roll)


import coremltools as ct

traced = torch.jit.trace(torch_model, img_tensor)


mlmodel = ct.convert(
    model=traced,
    source="pytorch",
    inputs=[ct.ImageType(name="input", shape=(1, 3, 128, 128), scale=1/255.0, bias=[0,0,0])],
    outputs=[ct.TensorType(name="output")]
)

# mlmodel = ct.models.MLModel("LandmarkDetectionModel.mlpackage")


img = Image.open(IMAGE_PATH).convert("RGB").resize((128, 128))

output = np.array(mlmodel.predict({"input": img})["output"]).flatten()

alpha_id, alpha_exp, pitch, yaw, roll, tX, tY, f = (
        output[0:219],
        output[219:258],
        output[258],
        output[259],
        output[260],
        output[261],
        output[262],
        output[263],
)

print("roll for coreml: ", roll)

mlmodel.save("LandmarkDetectionModel.mlpackage")



