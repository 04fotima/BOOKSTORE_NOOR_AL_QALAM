// CSRF token olish
function getCSRFToken() {
    const cookieValue = document.cookie
        .split('; ')
        .find(row => row.startsWith('csrftoken='))
        ?.split('=')[1];
    return cookieValue;
}

// Book Recommendation
document.getElementById('recommendForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const prompt = document.getElementById('prompt').value;
    
    const response = await fetch('/books/book-recommend/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        body: JSON.stringify({prompt})
    });

    const data = await response.json();
    document.getElementById('recommendationResult').innerText = data.recommendation;
});

// Book Review
document.getElementById('reviewForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const book_name = document.getElementById('bookName').value;

    const response = await fetch('/books/book-review/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        body: JSON.stringify({book_name})
    });

    const data = await response.json();
    document.getElementById('reviewResult').innerText = data.review;
});

// Reading Plan
document.getElementById('plannerForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const goals = document.getElementById('goals').value;

    const response = await fetch('/books/reading-plan/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken(),
        },
        body: JSON.stringify({goals})
    });

    const data = await response.json();
    document.getElementById('plannerResult').innerText = data.plan;
});
