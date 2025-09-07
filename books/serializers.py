# serializers.py
from rest_framework import serializers

class PromptSerializer(serializers.Serializer):
    age = serializers.CharField(max_length=10)
    interests = serializers.CharField(max_length=255)
    genre = serializers.CharField(max_length=100)


class DescriptionSerializer(serializers.Serializer):
    bookTitle = serializers.CharField(max_length=255)
    reviewText = serializers.CharField(max_length=1000)


class GoalSerializer(serializers.Serializer):
    bookPlanTitle = serializers.CharField(max_length=255)
    totalPages = serializers.IntegerField()
    days = serializers.IntegerField()
