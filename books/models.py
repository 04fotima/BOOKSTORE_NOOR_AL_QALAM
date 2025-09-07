
from django.db import models
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.conf import settings
import uuid
# from django.utils import timezone
# from django.contrib.auth.models import User
from django.contrib.auth.models import User
from django.db import models
from django.urls import reverse
from django.contrib.auth import get_user_model
from django.utils import timezone

class Book(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    isbn = models.CharField(max_length=20, unique=True, null=True, blank=True)
    rating = models.DecimalField(max_digits=2, decimal_places=1, default=0.0)
    cover = models.ImageField(upload_to="covers/", blank=True)
    date_creation = models.DateTimeField(auto_now_add=True)
    publisher = models.ForeignKey(
        get_user_model(),
        on_delete=models.SET_NULL,
        null=True,
    )
    genre = models.CharField(max_length=100, default="Unknown")
    description = models.TextField(default="No description available.", null=True, blank=True)
    pages = models.IntegerField(default=100, null=True, blank=True)
    published_date = models.DateField(null=True, blank=True)
    predicted_sales = models.IntegerField(null=True, blank=True)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def get_absolute_url(self):
        return reverse("book_detail", kwargs={"pk": self.pk})

    def __str__(self):
        return f"{self.title} - {self.predicted_sales if self.predicted_sales else 'Not Predicted'}"

class UserInterest(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    preferred_genres = models.CharField(max_length=255, blank=True)
    preferred_authors = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return f"Interest of {self.user.username}"




class BookCover(models.Model):
    """
    BookCover modeli kitob muqovasi, janri va yuklangan vaqtni saqlaydi.
    """
    image = models.ImageField(upload_to='covers/', verbose_name="Muqova Rasmi")
    genre = models.CharField(max_length=255, blank=True, null=True, verbose_name="Janr")
    uploaded_at = models.DateTimeField(auto_now_add=True, verbose_name="Yuklangan Vaqt")

    # def __str__(self):
    #     """
    #     Admin panelida va boshqa joylarda ko'rsatilishi.
    #     """
    #     return f"Cover - {self.genre if self.genre else 'Not Recognized'}"


User = get_user_model()  # <-- Use this for better custom user model support

class Review(models.Model):
    book = models.ForeignKey(
        'Book',   # Use string notation if 'Book' is declared after 'Review'
        on_delete=models.CASCADE,
        related_name="reviews",
    )
    author = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        # related_name='reviews'
    )
    content = models.TextField(default="No content provided.")
    sentiment = models.CharField(max_length=10, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # class Meta:
    #     ordering = ["-created_at"]

    # def __str__(self):
    #     return f"{self.author.username} - {self.book.title}"

    # def get_absolute_url(self):
    #     """
    #     Returns the absolute URL to the book detail page.
    #     """
    #     return reverse("book_detail", kwargs={"pk": self.book.pk})



# User = get_user_model()

# class Review(models.Model):
#     book = models.ForeignKey(
#         'Book',
#         on_delete=models.CASCADE,
#         related_name="reviews",
#     )
#     author = models.ForeignKey(
#         User,
#         on_delete=models.CASCADE,
#         related_name='reviews'
#     )
#     content = models.TextField()
#     sentiment = models.CharField(max_length=10, blank=True, null=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         ordering = ["-created_at"]

#     def __str__(self):
#         return f"{self.author.username} - {self.book.title}"

#     def get_absolute_url(self):
#         # Agar book_detail sahifasi 'book_detail' nomli URL name bo'lsa va bookning id si kerak bo'lsa:
#         return reverse("book_detail", kwargs={"pk": self.book.pk})

        
class Favorite(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.book.title}"



class CartItem(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    book = models.ForeignKey('Book', on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField(default=1)
    added_at = models.DateTimeField(auto_now_add=True)

    def subtotal(self):
        return self.book.price * self.quantity


class Order(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=20)
    card_number = models.CharField(max_length=20)
    created_at = models.DateTimeField(auto_now_add=True)
    is_paid = models.BooleanField(default=False)

    def total_price(self):
        return sum(item.subtotal() for item in self.items.all())


class OrderItem(models.Model):
    order = models.ForeignKey(Order, related_name='items', on_delete=models.CASCADE)
    book = models.ForeignKey(Book, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    price = models.DecimalField(max_digits=8, decimal_places=2)

    def subtotal(self):
        return self.price * self.quantity


class BookSalesPrediction(models.Model):
    book_title = models.CharField(max_length=255)  # Correct field name
    genre = models.CharField(max_length=50)
    published_date = models.DateField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    predicted_sales = models.IntegerField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.book_title} - {self.predicted_sales} copies"
