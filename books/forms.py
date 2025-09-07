# forms.py

from django import forms
from books.models import Review, Book, BookCover, BookSalesPrediction

GENRE_CHOICES = [
    ('Fantasy', 'Fantasy'),
    ('Science Fiction', 'Science Fiction'),
    ('Romance', 'Romance'),
    ('Mystery', 'Mystery'),
    ('Non-Fiction', 'Non-Fiction'),
]
class BookSalesForm(forms.ModelForm):
    genre = forms.ChoiceField(choices=GENRE_CHOICES)

    class Meta:
        model = BookSalesPrediction
        fields = ['genre', 'published_date', 'price']
        widgets = {
            'published_date': forms.DateInput(attrs={'type': 'date'}),
        }
  

class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ("content", "book")  # removed 'review' field as it does not exist in model
        # widgets = {
        #     'content': forms.Textarea(attrs={'placeholder': 'Fikringizni yozing...', 'class': 'form-control'})
        # }

class BookCoverForm(forms.ModelForm):
    """
    Kitob muqovasini yuklash uchun forma.
    """
    class Meta:
        model = BookCover
        fields = ['image']
        # widgets = {
        #     'image': forms.ClearableFileInput(attrs={'class': 'form-control'}),
        # }

class BookForm(forms.ModelForm):
    class Meta:
        model = Book
        fields = (
            "title",
            "author",
            "price",
            "isbn",
            "rating",
            "cover",
            "description",
            "pages",
            "published_date",
            "genre",
        )
