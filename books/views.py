from django.views.generic import ListView, DetailView, FormView, DeleteView, UpdateView, View
from django.views.generic.detail import SingleObjectMixin
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import redirect, render, get_object_or_404
from django.urls import reverse, reverse_lazy
from django.contrib import messages
from django.db.models import Q
from django.core.paginator import Paginator
from .models import Book, Review, Favorite
from .forms import ReviewForm, BookForm
from django.views.generic import ListView
from django.views import View
from .models import Book, CartItem, Order, OrderItem
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from .models import Book
from .serializers import PromptSerializer, DescriptionSerializer, GoalSerializer
from django.shortcuts import render
from django.views.generic import ListView, TemplateView
from django.db.models import Q
from .models import Book
from django.views import View
from django.views.generic import ListView, TemplateView
from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin

import os
import random
import numpy as np
# import joblib
from PIL import Image
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from textblob import TextBlob

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from textblob import TextBlob


from .models import (
    Book, UserInterest, Review, BookCover, BookSalesPrediction
)
from .forms import (
    ReviewForm, BookCoverForm, BookSalesForm
)


# === MODEL LOADING FOR SALES PREDICTION ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SALES_MODEL_PATH = os.path.join(BASE_DIR, 'books', 'model', 'model.pkl')

try:
    sales_model = joblib.load(SALES_MODEL_PATH)
    print("Sales prediction model loaded successfully.")
except FileNotFoundError:
    sales_model = None
    print(f"Sales prediction model not found at {SALES_MODEL_PATH}")

GENRE_MAPPING = {
    'Fantasy': 1,
    'Science Fiction': 2,
    'Romance': 3,
    'Mystery': 4,
    'Non-Fiction': 5
}
from django.views import View
from django.shortcuts import render
from .forms import BookSalesForm
from .models import BookSalesPrediction

class SalesPredictionView(View):
    template_name = 'books/sales_prediction.html'

    def get(self, request, *args, **kwargs):
        form = BookSalesForm()
        predictions = BookSalesPrediction.objects.all().order_by('-id')
        context = {
            'form': form,
            'prediction': None,
            'predictions': predictions,
        }
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        form = BookSalesForm(request.POST)
        prediction = None
        if form.is_valid():
            price = form.cleaned_data['price']

            # Get book_title if present, otherwise default to 'Unknown Title'
            book_title = form.cleaned_data.get('book_title', 'Unknown Title')

            # Simple sales prediction logic (example)
            prediction = price * 10

            # Create but don't save yet
            prediction_obj = form.save(commit=False)
            prediction_obj.predicted_sales = prediction
            prediction_obj.book_title = book_title  # Important to assign this explicitly
            prediction_obj.save()

        predictions = BookSalesPrediction.objects.all().order_by('-id')
        context = {
            'form': form,
            'prediction': prediction,
            'predictions': predictions,
        }
        return render(request, self.template_name, context)




# === MODEL LOADING FOR BOOK COVER GENRE CLASSIFICATION ===

GENRES = ['Fantasy', 'Science Fiction', 'Romance', 'Mystery', 'Non-Fiction']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


BOOK_COVER_MODEL_PATH = os.path.join(BASE_DIR, 'books', 'model', 'book_genre_classifier.pth')

try:
    if not os.path.exists(BOOK_COVER_MODEL_PATH) or os.path.getsize(BOOK_COVER_MODEL_PATH) == 0:
        print(f"Book cover classifier model file missing or empty at {BOOK_COVER_MODEL_PATH}. Using random predictions.")
        book_cover_model = None
    else:
        book_cover_model = SimpleCNN(num_classes=len(GENRES))
        book_cover_model.load_state_dict(torch.load(BOOK_COVER_MODEL_PATH, map_location=device))
        book_cover_model.to(device)
        book_cover_model.eval()
        print("Book cover classifier model loaded successfully.")
except Exception as e:
    print(f"Failed to load book cover model: {e}. Using random predictions.")
    book_cover_model = None


class BookCoverRecognitionView(View):
    template_name = 'books/cover_recognition.html'

    def get(self, request):
        form = BookCoverForm()
        covers = BookCover.objects.all().order_by('-uploaded_at')
        return render(request, self.template_name, {'form': form, 'covers': covers})

    def post(self, request):
        form = BookCoverForm(request.POST, request.FILES)
        if form.is_valid():
            book_cover = form.save()
            img_path = book_cover.image.path

            if book_cover_model is not None:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = book_cover_model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_index = torch.argmax(probabilities).item()
                    predicted_genre = GENRES[predicted_index]
                    print(f"Probabilities: {probabilities.cpu().numpy()}")
                    print(f"Predicted genre: {predicted_genre}")
            else:
                predicted_genre = random.choice(GENRES)
                print(f"Model not loaded. Random predicted genre: {predicted_genre}")

            book_cover.genre = predicted_genre
            book_cover.save()

            return redirect('cover_recognition')

        return render(request, self.template_name, {'form': form})


# === SENTIMENT ANALYSIS VIEW ===

class SentimentAnalysisView(View):
    template_name = 'books/sentiment_analysis.html'

    def get(self, request):
        form = ReviewForm()
        reviews = Review.objects.all().order_by('-created_at')
        return render(request, self.template_name, {'form': form, 'reviews': reviews})

    def post(self, request):
        form = ReviewForm(request.POST)
        if form.is_valid():
            review = form.save(commit=False)

            analysis = TextBlob(review.content)
            polarity = analysis.sentiment.polarity
            if polarity > 0:
                review.sentiment = "Ijobiy"
            elif polarity < 0:
                review.sentiment = "Salbiy"
            else:
                review.sentiment = "Neytral"

            review.author = request.user
            review.save()

            return redirect('sentiment_analysis')

        reviews = Review.objects.all().order_by('-created_at')
        return render(request, self.template_name, {'form': form, 'reviews': reviews})


# === HOME VIEW ===

class HomeView(TemplateView):
    template_name = 'books/home.html'


# === RECOMMENDATION VIEW ===

class RecommendationView(LoginRequiredMixin, ListView):
    model = Book
    template_name = 'books/recommend.html'
    context_object_name = 'recommended_books'

    def get_queryset(self):
        user = self.request.user
        try:
            interest = UserInterest.objects.get(user=user)
            genres = [g.strip() for g in interest.preferred_genres.split(',') if g.strip()]
            authors = [a.strip() for a in interest.preferred_authors.split(',') if a.strip()]
            
            # Genre va author bo'yicha filterlangan kitoblar
            recommended_books = Book.objects.filter(genre__in=genres) | Book.objects.filter(author__in=authors)
            
            # Reyting bo'yicha kamayish tartibida sortlash
            return recommended_books.distinct().order_by('-rating')
        
        except UserInterest.DoesNotExist:
            # Agar UserInterest bo'lmasa, eng yuqori reytingli 5 ta kitobni qaytarish
            return Book.objects.all().order_by('-rating')[:5]








# # 🔎 5️⃣ - Qidiruv optimallashtirish
# class SearchResultsListView(ListView):
#     model = Book
#     context_object_name = "book_list"
#     template_name = "books/navbar.html"
#     paginate_by = 10

#     def get_queryset(self):
#         query = self.request.GET.get("q", None)
#         if query:
#             # Fuzzy Search orqali qidiruv
#             titles = Book.objects.values_list('title', flat=True)
#             results = process.extract(query, titles, limit=10)
#             books = Book.objects.filter(title__in=[result[0] for result in results])
#             return books
#         return Book.objects.none()

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context["query"] = self.request.GET.get("q")
#         return context


# # 🔄 6️⃣ - Saralash
# class BookListView(ListView):
#     model = Book
#     context_object_name = "book_list"
#     template_name = "books/navbar.html"
#     paginate_by = 24

#     def get_queryset(self):
#         sort_option = self.request.GET.get("sort", "")
#         queryset = Book.objects.all()

#         # Saralash mezonlari
#         if sort_option == "price_asc":
#             queryset = queryset.order_by("price")
#         elif sort_option == "price_desc":
#             queryset = queryset.order_by("-price")
#         elif sort_option == "rating":
#             queryset = queryset.order_by("-rating")
#         elif sort_option == "newest":
#             queryset = queryset.order_by("-date_creation")
#         elif sort_option == "oldest":
#             queryset = queryset.order_by("date_creation")
#         elif sort_option == "author":
#             queryset = queryset.order_by("author")
#         elif sort_option == "title":
#             queryset = queryset.order_by("title")

#         return queryset


















# Model yuklash (bir marta yuklanadi, GPT-2 o'rniga generatsiya uchun DRF ishlatilmoqda)

# ✅ Book Recommender API
@method_decorator(csrf_exempt, name='dispatch')
class BookRecommenderAPIView(APIView):
    def post(self, request):
        serializer = PromptSerializer(data=request.data)
        if serializer.is_valid():
            age = serializer.data.get('age')
            interests = serializer.data.get('interests')
            genre = serializer.data.get('genre')

            # Book filtering logic
            recommendations = Book.objects.filter(
                genre__icontains=genre,
                description__icontains=interests
            )[:5]

            # Extract book titles
            books_list = [book.title for book in recommendations]
            message = "Recommended Books: " + ", ".join(books_list) if books_list else "No recommendations found."
            
            return Response({"recommendations": message}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ✅ Book Review API
@method_decorator(csrf_exempt, name='dispatch')
class BookReviewAPIView(APIView):
    def post(self, request):
        serializer = DescriptionSerializer(data=request.data)
        if serializer.is_valid():
            book_title = serializer.data.get('bookTitle')
            review_text = serializer.data.get('reviewText')

            # Simple response simulation
            message = f"Your review for '{book_title}' has been successfully submitted."
            
            return Response({"message": message}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ✅ Reading Planner API
@method_decorator(csrf_exempt, name='dispatch')
class ReadingPlannerAPIView(APIView):
    def post(self, request):
        serializer = GoalSerializer(data=request.data)
        if serializer.is_valid():
            book_plan_title = serializer.data.get('bookPlanTitle')
            total_pages = int(serializer.data.get('totalPages'))
            days = int(serializer.data.get('days'))

            # Calculate pages per day
            if days > 0:
                pages_per_day = total_pages // days
                message = f"To finish '{book_plan_title}' in {days} days, you need to read about {pages_per_day} pages per day."
            else:
                message = "Days must be greater than 0."

            return Response({"plan": message}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class CartPageView(LoginRequiredMixin, ListView):
    model = CartItem
    template_name = 'books/cart.html'
    context_object_name = 'cart_items'

    def get_queryset(self):
        return CartItem.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        cart_items = context['cart_items']
        context['total'] = sum(item.subtotal() for item in cart_items)
        return context


class AddToCartView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        book_id = kwargs.get('book_id')
        book = get_object_or_404(Book, id=book_id)
        cart_item, created = CartItem.objects.get_or_create(user=request.user, book=book)

        if not created:
            cart_item.quantity += 1
            cart_item.save()

        return redirect('cart_page')


class CheckoutView(LoginRequiredMixin, View):
    def post(self, request, *args, **kwargs):
        cart_items = CartItem.objects.filter(user=request.user)
        if not cart_items.exists():
            return redirect('cart_page')

        order = Order.objects.create(
            user=request.user,
            full_name=request.POST.get('full_name'),
            phone_number=request.POST.get('phone_number'),
            card_number=request.POST.get('card_number'),
            is_paid=True  # Agar real payment bo'lsa, bu False bo'ladi
        )

        for item in cart_items:
            OrderItem.objects.create(
                order=order,
                book=item.book,
                quantity=item.quantity,
                price=item.book.price
            )

        cart_items.delete()

        return render(request, 'books/success.html', {'order': order})

    def get(self, request, *args, **kwargs):
        return redirect('cart_page')



class FavoriteListView(LoginRequiredMixin, ListView):
    model = Favorite
    template_name = 'favorite_list.html'
    context_object_name = 'favorites'

    def get_queryset(self):
        # Ensure the user is authenticated before filtering the favorites
        if self.request.user.is_authenticated:
            return Favorite.objects.filter(user=self.request.user)
        else:
            # Return an empty queryset or redirect if user is not authenticated
            return Favorite.objects.none()

# Add book to favorites

class AddToFavoritesView(View):
    def post(self, request, book_id):
        # Ensure the user is logged in
        if not request.user.is_authenticated:
            messages.error(request, "You must be logged in to add to favorites.")
            return redirect('favorite_list')

        book = get_object_or_404(Book, id=book_id)

        # Ensure the user hasn't already added this book to their favorites
        if Favorite.objects.filter(user=request.user, book=book).exists():
            messages.info(request, "This book is already in your favorites.")
        else:
            Favorite.objects.create(user=request.user, book=book)
            messages.success(request, "Book added to your favorites! ❤️")

        return redirect('favorite_list')  # Redirect to favorites list page

class RemoveFromFavoritesView(LoginRequiredMixin, View):
    def get(self, request, favorite_id):
        # Ensure the user is logged in, LoginRequiredMixin handles that
        favorite = get_object_or_404(Favorite, id=favorite_id)

        # Check if the favorite belongs to the logged-in user
        if favorite.user == request.user:
            favorite.delete()  # Remove the favorite from the database
            messages.success(request, 'Book has been removed from favorites! ❌')
        else:
            messages.error(request, 'You cannot remove this book from favorites!')

        return redirect('favorite_list')
    
class BookListView(ListView):
    paginate_by = 24
    model = Book
    context_object_name = "book_list"
    template_name = "books/book_list.html"
    # queryset = Book.objects.all().order_by("-date_creation")
    def get_queryset(self):
        sort_option = self.request.GET.get("sort", "")
        queryset = Book.objects.all()

        # Sorting logic
        if sort_option == "price_asc":
            queryset = queryset.order_by("price")  # Sort by price (Low to High)
        elif sort_option == "price_desc":
            queryset = queryset.order_by("-price")  # Sort by price (High to Low)
        elif sort_option == "rating":
            queryset = queryset.order_by("-rating")  # Sort by rating (High to Low)
        elif sort_option == "newest":
            queryset = queryset.order_by("-date_creation")  # Sort by newest first
        elif sort_option == "oldest":
            queryset = queryset.order_by("date_creation")  # Sort by oldest first
        elif sort_option == "author":
            queryset = queryset.order_by("author")  # Sort by author (A-Z)
        elif sort_option == "title":
            queryset = queryset.order_by("title")  # Sort by title (A-Z)

        return queryset

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["sort"] = self.request.GET.get("sort", "")  # Pass the current sort option to the template
        return context

class ReviewGet(DetailView):
    model = Book
    context_object_name = "book"
    template_name = "books/book_detail.html"
    queryset = Book.objects.all().prefetch_related(
        "reviews__author",
    )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        context["form"] = self.get_review_form()
        context["page_obj"] = self.get_review_pagination(context)
        return context

    def get_review_form(self):
        return ReviewForm()

    def get_review_pagination(self, context):
        """returns pagination for reviews section"""
        reviews_list = Review.objects.filter(book=context["book"])
        paginator = Paginator(reviews_list, 5)
        page = self.request.GET.get("page")
        page_obj = paginator.get_page(page)
        return page_obj


class ReviewPost(SingleObjectMixin, FormView):
    model = Book
    context_object_name = "book"
    form_class = ReviewForm
    template_name = "books/book_detail.html"
    queryset = Book.objects.all().prefetch_related(
        "reviews__author",
    )

    def post(self, request, *args, **kwargs):
        self.object = self.get_object()
        noti_msj = f"{self.request.user} Review was added successfully"
        messages.success(self.request, noti_msj)
        return super().post(request, *args, **kwargs)

    def form_valid(self, form):
        review = form.save(commit=False)
        review.book = self.object
        review.author = self.request.user
        review.save()
        return super().form_valid(form)

    def get_success_url(self):
        book = self.get_object()
        return reverse("book_detail", kwargs={"pk": book.pk})


class BookDetailView(View):
    def get(self, request, *args, **kwargs):
        view = ReviewGet.as_view()
        return view(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        view = ReviewPost.as_view()
        return view(request, *args, **kwargs)


class SearchResultsListView(ListView):
    paginate_by = 10
    model = Book
    context_object_name = "book_list"
    template_name = "books/search_results.html"

    def get_queryset(self):
        query = self.request.GET.get("q", None)
        if query:
            return Book.objects.filter(
                Q(title__icontains=query) | Q(author__icontains=query)
            ).order_by("-date_creation")
        return []

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["query"] = self.request.GET.get("q")
        return context


class MyBooksView(LoginRequiredMixin, ListView):
    paginate_by = 10
    model = Book
    context_object_name = "my_books"
    template_name = "books/book_account.html"

    def get_queryset(self):
        user = self.request.user
        return Book.objects.filter(publisher=user).order_by("-date_creation")

    def dispatch(self, request, *args, **kwargs):
        """Will send a notification in case user is not logged"""
        if not request.user.is_authenticated:
            messages.error(self.request, "You need to log in")
        return super().dispatch(request, *args, **kwargs)


class NewBookView(LoginRequiredMixin, FormView):
    form_class = BookForm
    template_name = "books/book_new.html"

    def dispatch(self, request, *args, **kwargs):
        """Will send a notification in case user is not logged"""
        if not request.user.is_authenticated:
            messages.error(self.request, "You need to log in to published a book")
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        book = form.save(commit=False)
        book.publisher = self.request.user
        book.save()
        self.success_url = reverse("book_detail", kwargs={"pk": book.id})

        if form.is_valid():
            messages.success(self.request, "Book Published Correctly")

        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(self.request, "Error Publishing Book")
        return super().form_invalid(form)


class DeleteBookView(LoginRequiredMixin, UserPassesTestMixin, DeleteView):
    model = Book
    template_name = "books/book_confirm_delete.html"
    success_url = reverse_lazy("book_account")

    def dispatch(self, request, *args, **kwargs):
        """Will send a notification in case user is not logged"""
        if not request.user.is_authenticated:
            messages.error(
                self.request,
                "You need to log in to delete a book, and be its publisher",
            )
        return super().dispatch(request, *args, **kwargs)

    def test_func(self):
        """Can delete if user is who published the book"""
        return self.request.user == self.get_object().publisher

    def handle_no_permission(self):
        messages.error(self.request, "You cannot delete other's books")
        return redirect(self.get_object().get_absolute_url())

    def get(self, request, *args, **kwargs):
        messages.warning(self.request, "Be Carefull You're going to delete this book")
        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        messages.info(self.request, f"Book '{self.get_object()}' Delete")
        return super().post(request, *args, **kwargs)


class UpdateBookView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    model = Book
    form_class = BookForm
    template_name = "books/book_update.html"

    def dispatch(self, request, *args, **kwargs):
        """Will send a notification in case user is not logged"""
        if not request.user.is_authenticated:
            messages.error(
                self.request,
                "You need to log in to delete a book, and be its publisher",
            )
        return super().dispatch(request, *args, **kwargs)

    def test_func(self):
        """Can update if user is who published the book"""
        return self.request.user == self.get_object().publisher

    def handle_no_permission(self):
        messages.error(self.request, "You cannot update other's books")
        return redirect(self.get_object().get_absolute_url())

    def get(self, request, *args, **kwargs):
        messages.info(self.request, "Updating Book")
        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        messages.success(self.request, f"Book '{self.get_object()}' Updated")
        return super().post(request, *args, **kwargs)

    def get_success_url(self):
        return self.get_object().get_absolute_url()

