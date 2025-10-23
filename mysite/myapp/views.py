from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
from .model import VideoModel
from .preprocess_video import preprocess_video
import torch

model = VideoModel(hidden_size=32, num_layers=1, dropout=0.5)

model.load_state_dict(torch.load("myapp/model/cnn_gru.pth", map_location=torch.device('cpu')))
model.eval() 


def upload_video(request):
    result = None
    if request.method == 'POST' and request.FILES['video']:
        video_file = request.FILES['video']
        fs = FileSystemStorage()
        filename = fs.save(video_file.name, video_file)
        file_path = fs.path(filename)

        video_tensor = preprocess_video(file_path)  

        with torch.no_grad():
            output = model(video_tensor)
            predicted = torch.sigmoid(output)

        result = " Theft detected!" if predicted.item() >=0.5 else " Normal activity"

    return render(request, 'upload.html', {'result': result})

