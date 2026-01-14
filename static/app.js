// Use relative URL to avoid CORS issues when served from same origin
const API_BASE_URL = window.location.origin;

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('queryForm');
    const queryInput = document.getElementById('query');
    const hotelFilterInput = document.getElementById('hotelFilter');
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsDiv = document.getElementById('results');
    const answerDiv = document.getElementById('answer');
    const sourcesSection = document.getElementById('sourcesSection');
    const sourcesDiv = document.getElementById('sources');
    const errorDiv = document.getElementById('error');
    const relevantBadge = document.getElementById('relevantBadge');

    if (!form) {
        console.error('Form not found!');
        return;
    }

    function displayResults(data) {
        // Display answer
        answerDiv.textContent = data.answer;
        
        // Display relevance badge
        if (data.relevant) {
            relevantBadge.textContent = '✓ Relevant';
            relevantBadge.className = 'badge relevant';
        } else {
            relevantBadge.textContent = '✗ Not Relevant';
            relevantBadge.className = 'badge not-relevant';
        }
        
        // Display sources if available
        if (data.sources && data.sources.length > 0) {
            sourcesDiv.innerHTML = '';
            data.sources.forEach((source, index) => {
                const sourceItem = document.createElement('div');
                sourceItem.className = 'source-item';
                sourceItem.innerHTML = `<strong>Source ${index + 1}:</strong> ${escapeHtml(source)}`;
                sourcesDiv.appendChild(sourceItem);
            });
            sourcesSection.style.display = 'block';
        } else {
            sourcesSection.style.display = 'none';
        }
        
        // Show results
        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Hide previous results and errors
        resultsDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        
        // Show loading state
        submitBtn.disabled = true;
        submitText.textContent = 'Searching...';
        loadingSpinner.style.display = 'inline-block';
        
        const query = queryInput.value.trim();
        const hotelFilter = hotelFilterInput.value.trim() || null;
        
        if (!query) {
            showError('Please enter a question.');
            submitBtn.disabled = false;
            submitText.textContent = 'Search Reviews';
            loadingSpinner.style.display = 'none';
            return;
        }
        
        try {
            console.log('Sending request to:', `${API_BASE_URL}/query`);
            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    hotel_filter: hotelFilter,
                }),
            });
            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }
            
            const data = await response.json();
            console.log('Response data:', data);
            
            if (!data || !data.answer) {
                throw new Error('Invalid response format from server');
            }
            
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            const errorMsg = error.message || 'Unknown error occurred';
            showError(`Failed to get answer: ${errorMsg}. Make sure the API server is running and the vector store is populated.`);
        } finally {
            // Reset button state
            submitBtn.disabled = false;
            submitText.textContent = 'Search Reviews';
            loadingSpinner.style.display = 'none';
        }
    });
});
