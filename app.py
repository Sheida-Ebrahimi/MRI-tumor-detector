from flask import Flask, render_template, request
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(save_gradient)

    def __call__(self, x):
        output = self.model(x)

        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= cam.max() + 1e-8

        return cam.cpu().numpy()


def predict(img_pil):
    img_pil = img_pil.convert("RGB")
    x = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits)[0].item()

    return prob

def generate_heatmap(img_pil, save_path):
    img = img_pil.convert("RGB")
    x = transform(img).unsqueeze(0)

    # Grad-CAM mask
    cam = gradcam(x)
    cam = cv2.resize(cam, (img.width, img.height))

    # Normalize heatmap
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convert PIL → numpy
    orig = np.array(img)
    orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

    # Overlay
    overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)


app = Flask(__name__)

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

state_dict = torch.load("brain_tumor_resnet18.pt", map_location="cpu")
model.load_state_dict(state_dict)
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            filename = file.filename

            upload_path = os.path.join("static", "uploads", filename)
            os.makedirs("static/uploads", exist_ok=True)
            file.save(upload_path)

            img = Image.open(upload_path)
            tumor_prob = predict(img)

            if tumor_prob >= 0.5:
                result = f"TUMOR DETECTED (confidence: {tumor_prob:.2f})"
            else:
                result = f"NO TUMOR (confidence: {(1 - tumor_prob):.2f})"

            # create heatmap path
            heatmap_path = os.path.join("static", "outputs", filename)
            os.makedirs("static/outputs", exist_ok=True)

            # generate GradCAM heatmap
            generate_heatmap(img, heatmap_path)

            image_url = upload_path
            heatmap_url = heatmap_path


        return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        heatmap_url=heatmap_url
    )



if __name__ == "__main__":
    app.run(debug=True)
