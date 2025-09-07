from django.urls import path

from .views import (
    BookListView,
    BookDetailView,
    SearchResultsListView,
    MyBooksView,
    NewBookView,
    DeleteBookView,
    UpdateBookView,
    FavoriteListView,
    AddToFavoritesView,
    RemoveFromFavoritesView,
    CartPageView, AddToCartView, CheckoutView, BookRecommenderAPIView, BookReviewAPIView, ReadingPlannerAPIView,
    RecommendationView,  BookCoverRecognitionView,
    SentimentAnalysisView,
    RecommendationView,
    HomeView, SalesPredictionView
    

)
urlpatterns = [


    path('', BookListView.as_view(), name='book_list'), 
    path('', HomeView.as_view(), name='home'),
    path("<uuid:pk>/", BookDetailView.as_view(), name="book_detail"),
    path("search/", SearchResultsListView.as_view(), name="search_results"),
    path("account/", MyBooksView.as_view(), name="book_account"),
    path("new/", NewBookView.as_view(), name="book_new"),
    path("delete/<uuid:pk>/", DeleteBookView.as_view(), name="book_delete"),
    path("update/<uuid:pk>/", UpdateBookView.as_view(), name="book_update"),
    path("favorite_list/", FavoriteListView.as_view(), name="favorite_list"),
    path("favorite_list/add/<uuid:book_id>/", AddToFavoritesView.as_view(), name="add_to_favorites"),
    path("favorite_list/remove/<uuid:favorite_id>/", RemoveFromFavoritesView.as_view(), name="remove_from_favorites"),
    path('cart/', CartPageView.as_view(), name='cart_page'),
    path('cart/add/<uuid:book_id>/', AddToCartView.as_view(), name='add_to_cart'),
    path('checkout/', CheckoutView.as_view(), name='checkout'),
    path('book-recommend/', BookRecommenderAPIView.as_view(), name='book-recommend'),
    path('book-review/', BookReviewAPIView.as_view(), name='book-review'),
    path('reading-plan/', ReadingPlannerAPIView.as_view(), name='reading-plan'),
    
    path('cover_recognition/', BookCoverRecognitionView.as_view(), name='cover_recognition'),
    path('sentiment-analysis/', SentimentAnalysisView.as_view(), name='sentiment_analysis'),
    path('recommend/', RecommendationView.as_view(), name='recommend'),
    path('sales-prediction/', SalesPredictionView.as_view(), name='sales_prediction'),

]
