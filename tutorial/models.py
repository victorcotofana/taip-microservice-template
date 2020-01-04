import json

from rest_framework.views import APIView
from rest_framework.response import Response

from tutorial.module.detection import load_and_predict

class ApiView(APIView):
    def get(self, request):
        return Response({'some': 'data'})

    def post(self, request):
        img_base64 = request.data['base64img']
        result = load_and_predict(img_base64)
        return Response(result)