import torch
from torchviz import make_dot
from model import ImprovedSpectrogramClassifier

num_classes = 50
model = ImprovedSpectrogramClassifier(num_classes=num_classes)

model.load_state_dict(torch.load("artifacts/fix_model72.pth", map_location='cpu'))

model.eval()

x = torch.randn(1, 1, 128, 128, requires_grad=True)
y = model(x)
print(y)

dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_graph", format="png")  # создаст model_graph.png