// Application State
const state = {
    currentPage: 'home',
    isLoading: false,
    lastAnalysis: null,
    history: [
        {
            id: 1,
            timestamp: '2025-06-13T19:10:00Z',
            text: 'Банк объявил о новых тарифах по ипотечному кредитованию',
            toxicity_score: 0.02,
            is_toxic: false,
            rating: 'BBB',
            processing_time: 0.31
        },
        {
            id: 2,
            timestamp: '2025-06-13T19:05:00Z',
            text: 'Это полный провал вашей политики! Абсолютно неприемлемо!',
            toxicity_score: 0.89,
            is_toxic: true,
            rating: 'C',
            processing_time: 0.28
        },
        {
            id: 3,
            timestamp: '2025-06-13T19:00:00Z',
            text: 'Превосходные результаты работы в третьем квартале',
            toxicity_score: 0.01,
            is_toxic: false,
            rating: 'AA',
            processing_time: 0.29
        }
    ]
};

// API Configuration
const API_CONFIG = {
    baseUrl: 'http://localhost:8000',
    endpoints: {
        classify: '/api/v1/classification/classify-text'
    },
    timeout: 10000 // 10 seconds timeout
};

// Chart configurations
const chartColors = ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5', '#5D878F', '#DB4545', '#D2BA4C'];

// DOM Elements
const elements = {
    pages: document.querySelectorAll('.page'),
    navLinks: document.querySelectorAll('.nav-link'),
    textInput: null,
    charCounter: null,
    analyzeBtn: null,
    resultsSection: null,
    loadingOverlay: null,
    toastContainer: null
};

// Initialize Application
document.addEventListener('DOMContentLoaded', function() {
    initializeDOMElements();
    initializeApp();
    initializeEventListeners();
    initializeRouting();
});

function initializeDOMElements() {
    elements.textInput = document.getElementById('text-input');
    elements.charCounter = document.getElementById('char-counter');
    elements.analyzeBtn = document.getElementById('analyze-btn');
    elements.resultsSection = document.getElementById('results-section');
    elements.loadingOverlay = document.getElementById('loading-overlay');
    elements.toastContainer = document.getElementById('toast-container');
}

function initializeApp() {
    // Set initial route
    const hash = window.location.hash.replace('#/', '') || 'home';
    navigateTo(hash);
    
    // Add typing animation to hero title
    const heroTitle = document.querySelector('.hero-title');
    if (heroTitle) {
        setTimeout(() => {
            heroTitle.style.borderRight = 'none';
        }, 3000);
    }
}

function initializeEventListeners() {
    // Navigation
    elements.navLinks.forEach(link => {
        link.addEventListener('click', handleNavigation);
    });

    // Text input character counter
    if (elements.textInput) {
        elements.textInput.addEventListener('input', updateCharCounter);
    }

    // Analyze button
    if (elements.analyzeBtn) {
        elements.analyzeBtn.addEventListener('click', handleAnalyze);
    }

    // Window resize for responsive charts
    window.addEventListener('resize', debounce(handleResize, 250));
}

function initializeRouting() {
    window.addEventListener('hashchange', () => {
        const hash = window.location.hash.replace('#/', '') || 'home';
        navigateTo(hash);
    });
}

function handleNavigation(e) {
    e.preventDefault();
    const route = e.currentTarget.dataset.route;
    navigateTo(route);
}

function navigateTo(route) {
    // Update URL
    window.location.hash = `#/${route}`;
    
    // Update active nav link
    elements.navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.dataset.route === route) {
            link.classList.add('active');
        }
    });

    // Show active page
    elements.pages.forEach(page => {
        page.classList.remove('active');
        const pageId = `${route}-page`;
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add('active');
        }
    });

    state.currentPage = route;
    
    // Initialize page-specific content
    setTimeout(() => {
        if (route === 'dashboard') {
            initializeDashboard();
        } else if (route === 'history') {
            initializeHistory();
        }
    }, 100);
}

function updateCharCounter() {
    if (!elements.textInput || !elements.charCounter) return;
    
    const text = elements.textInput.value;
    const charCount = text.length;
    elements.charCounter.textContent = charCount;
    
    // Update analyze button state
    const isValid = charCount >= 10;
    if (elements.analyzeBtn) {
        elements.analyzeBtn.disabled = !isValid;
    }
    
    // Add visual feedback
    if (charCount > 0 && charCount < 10) {
        elements.charCounter.style.color = '#ff6b6b';
    } else if (charCount >= 10) {
        elements.charCounter.style.color = '#4ecdc4';
    } else {
        elements.charCounter.style.color = 'rgba(255, 255, 255, 0.6)';
    }
}

async function handleAnalyze() {
    if (!elements.textInput) return;
    
    const text = elements.textInput.value.trim();
    
    if (text.length < 10) {
        showToast('Минимальная длина текста: 10 символов', 'error');
        return;
    }

    setLoadingState(true);
    
    try {
        // Try to classify text with API
        const result = await classifyText(text);
        displayResults(result);
        addToHistory(text, result);
        showToast('Анализ завершен успешно!', 'success');
        
    } catch (error) {
        console.warn('API не доступен, используем демо-данные:', error);
        
        // Generate demo results
        const demoResult = generateDemoResult(text);
        displayResults(demoResult);
        addToHistory(text, demoResult);
        showToast('Использованы демо-данные (API недоступен)', 'info');
        
    } finally {
        setLoadingState(false);
    }
}

async function classifyText(text) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_CONFIG.timeout);
    
    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('include_toxicity', 'true');
        formData.append('include_rating', 'true');

        const response = await fetch(`${API_CONFIG.baseUrl}${API_CONFIG.endpoints.classify}`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        throw error;
    }
}

function generateDemoResult(text) {
    const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
    const toxicityScore = Math.random() * 0.6 + 0.05; // 5-65%
    const isToxic = toxicityScore > 0.5;
    const ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'C'];
    const selectedRating = ratings[Math.floor(Math.random() * ratings.length)];
    
    // Generate probabilities that sum to 1
    const probabilities = {};
    let remaining = 1.0;
    ratings.forEach((rating, index) => {
        if (index === ratings.length - 1) {
            probabilities[rating] = remaining;
        } else {
            const value = remaining * Math.random() * 0.5;
            probabilities[rating] = value;
            remaining -= value;
        }
    });
    
    // Boost the selected rating
    probabilities[selectedRating] = Math.max(probabilities[selectedRating], 0.15 + Math.random() * 0.25);
    
    return {
        original_text_length: text.length,
        word_count: wordCount,
        toxicity: {
            toxicity_score: toxicityScore,
            is_toxic: isToxic,
            confidence: Math.random() * 0.3 + 0.7,
            probabilities: {
                non_toxic: 1 - toxicityScore,
                toxic: toxicityScore
            },
            processing_time: Math.random() * 0.4 + 0.1,
            from_cache: Math.random() > 0.7,
            model_name: 's-nlp/russian_toxicity_classifier'
        },
        rating: {
            category: selectedRating,
            confidence: probabilities[selectedRating],
            probabilities: probabilities,
            processing_time: Math.random() * 0.08 + 0.02,
            from_cache: Math.random() > 0.8,
            model_name: 'cointegrated/rubert-tiny2'
        },
        total_processing_time: Math.random() * 0.5 + 0.15
    };
}

function addToHistory(text, result) {
    const historyEntry = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        text: text.substring(0, 50) + (text.length > 50 ? '...' : ''),
        toxicity_score: result.toxicity.toxicity_score,
        is_toxic: result.toxicity.is_toxic,
        rating: result.rating.category,
        processing_time: result.total_processing_time
    };
    
    state.history.unshift(historyEntry);
    state.lastAnalysis = result;
    
    // Update history table if currently visible
    if (state.currentPage === 'history') {
        initializeHistory();
    }
}

function displayResults(result) {
    // Update text statistics
    animateNumber('text-length', result.original_text_length);
    animateNumber('word-count', result.word_count);
    
    // Update toxicity analysis
    displayToxicityResults(result.toxicity);
    
    // Update rating classification
    displayRatingResults(result.rating);
    
    // Update performance metrics
    const processingTimeEl = document.getElementById('processing-time');
    const cacheStatusEl = document.getElementById('cache-status');
    
    if (processingTimeEl) {
        processingTimeEl.textContent = `${(result.total_processing_time * 1000).toFixed(0)} мс`;
    }
    if (cacheStatusEl) {
        cacheStatusEl.textContent = result.toxicity.from_cache ? 'Да' : 'Нет';
    }
    
    // Show results section
    if (elements.resultsSection) {
        elements.resultsSection.classList.remove('hidden');
        elements.resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
}

function displayToxicityResults(toxicity) {
    const percentage = Math.round(toxicity.toxicity_score * 100);
    const confidence = Math.round(toxicity.confidence * 100);
    
    // Update text elements
    const percentageEl = document.getElementById('toxicity-percentage');
    const confidenceEl = document.getElementById('toxicity-confidence');
    const modelEl = document.getElementById('toxicity-model');
    const statusElement = document.getElementById('toxicity-status');
    
    if (percentageEl) percentageEl.textContent = `${percentage}%`;
    if (confidenceEl) confidenceEl.textContent = `${confidence}%`;
    if (modelEl) modelEl.textContent = toxicity.model_name;
    
    if (statusElement) {
        if (toxicity.is_toxic) {
            statusElement.textContent = 'Токсичен';
            statusElement.className = 'toxicity-status toxic';
        } else {
            statusElement.textContent = 'Не токсичен';
            statusElement.className = 'toxicity-status non-toxic';
        }
    }
    
    // Create toxicity donut chart
    setTimeout(() => createToxicityChart(toxicity.toxicity_score), 200);
}

function displayRatingResults(rating) {
    const confidence = Math.round(rating.confidence * 100);
    
    // Update main rating
    const mainRatingEl = document.getElementById('main-rating');
    const ratingConfidenceEl = document.getElementById('rating-confidence');
    
    if (mainRatingEl) mainRatingEl.textContent = rating.category;
    if (ratingConfidenceEl) ratingConfidenceEl.textContent = `${confidence}%`;
    
    // Create rating bar chart
    setTimeout(() => createRatingChart(rating.probabilities), 300);
}

function createToxicityChart(score) {
    const canvas = document.getElementById('toxicity-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear previous chart
    if (window.toxicityChart) {
        window.toxicityChart.destroy();
    }
    
    const percentage = score * 100;
    const color = percentage > 70 ? '#ff6b6b' : percentage > 30 ? '#ffa500' : '#4ecdc4';
    
    window.toxicityChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: [color, 'rgba(255, 255, 255, 0.1)'],
                borderWidth: 0,
                cutout: '70%'
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            animation: {
                animateRotate: true,
                duration: 1500
            }
        }
    });
}

function createRatingChart(probabilities) {
    const canvas = document.getElementById('rating-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear previous chart
    if (window.ratingChart) {
        window.ratingChart.destroy();
    }
    
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities).map(v => v * 100);
    
    window.ratingChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: chartColors.slice(0, labels.length),
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: Math.max(...data) * 1.2,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: function(value) {
                            return value.toFixed(1) + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutBounce'
            }
        }
    });
}

function initializeDashboard() {
    // Skip if already initialized or not on dashboard
    if (window.dashboardInitialized || state.currentPage !== 'dashboard') return;
    
    // Mock data for dashboard
    const mockData = {
        dailyStats: [
            { date: '2025-06-07', requests: 1200, avg_toxicity: 0.12 },
            { date: '2025-06-08', requests: 1800, avg_toxicity: 0.18 },
            { date: '2025-06-09', requests: 2100, avg_toxicity: 0.23 },
            { date: '2025-06-10', requests: 1900, avg_toxicity: 0.19 },
            { date: '2025-06-11', requests: 2300, avg_toxicity: 0.31 },
            { date: '2025-06-12', requests: 2800, avg_toxicity: 0.28 },
            { date: '2025-06-13', requests: 3300, avg_toxicity: 0.26 }
        ],
        ratingDistribution: {
            AAA: 15, AA: 23, A: 18, BBB: 28, BB: 12, B: 3, C: 1
        }
    };
    
    // Create charts with delay to ensure DOM is ready
    setTimeout(() => {
        createActivityChart(mockData.dailyStats);
        createDistributionChart(mockData.ratingDistribution);
        window.dashboardInitialized = true;
    }, 300);
}

function createActivityChart(data) {
    const canvas = document.getElementById('activity-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear previous chart
    if (window.activityChart) {
        window.activityChart.destroy();
    }
    
    window.activityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => new Date(d.date).toLocaleDateString('ru-RU', { month: 'short', day: 'numeric' })),
            datasets: [{
                label: 'Запросы',
                data: data.map(d => d.requests),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: 'rgba(255, 255, 255, 0.7)' }
                }
            }
        }
    });
}

function createDistributionChart(data) {
    const canvas = document.getElementById('distribution-chart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    // Clear previous chart
    if (window.distributionChart) {
        window.distributionChart.destroy();
    }
    
    window.distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(data),
            datasets: [{
                data: Object.values(data),
                backgroundColor: chartColors,
                borderWidth: 2,
                borderColor: 'rgba(255, 255, 255, 0.1)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: 'rgba(255, 255, 255, 0.7)' }
                }
            }
        }
    });
}

function initializeHistory() {
    const tbody = document.getElementById('history-tbody');
    if (!tbody) return;
    
    tbody.innerHTML = state.history.map(entry => `
        <tr>
            <td>${new Date(entry.timestamp).toLocaleString('ru-RU')}</td>
            <td>${entry.text}</td>
            <td class="${entry.is_toxic ? 'toxicity-high' : 'toxicity-low'}">
                ${(entry.toxicity_score * 100).toFixed(1)}%
            </td>
            <td><span class="rating-badge">${entry.rating}</span></td>
            <td>${(entry.processing_time * 1000).toFixed(0)} мс</td>
            <td>
                <button class="btn--details" onclick="showDetails(${entry.id})">
                    Подробнее
                </button>
            </td>
        </tr>
    `).join('');
}

function showDetails(id) {
    const entry = state.history.find(h => h.id === id);
    if (entry) {
        showToast(`Детали записи #${id}: ${entry.text}`, 'info');
    }
}

function setLoadingState(isLoading) {
    state.isLoading = isLoading;
    
    if (elements.loadingOverlay) {
        if (isLoading) {
            elements.loadingOverlay.classList.remove('hidden');
        } else {
            elements.loadingOverlay.classList.add('hidden');
        }
    }
    
    if (elements.analyzeBtn) {
        const btnText = elements.analyzeBtn.querySelector('.btn-text');
        const btnLoader = elements.analyzeBtn.querySelector('.btn-loader');
        
        if (isLoading) {
            elements.analyzeBtn.disabled = true;
            if (btnText) btnText.textContent = 'Анализируем...';
            if (btnLoader) btnLoader.classList.remove('hidden');
        } else {
            elements.analyzeBtn.disabled = false;
            if (btnText) btnText.textContent = 'Анализировать текст';
            if (btnLoader) btnLoader.classList.add('hidden');
        }
    }
}

function showToast(message, type = 'info') {
    if (!elements.toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        toast.style.transform = 'translateX(100%)';
        toast.style.opacity = '0';
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

function animateNumber(elementId, targetValue) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const startValue = 0;
    const duration = 1000;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const easeOutQuart = 1 - Math.pow(1 - progress, 4);
        const currentValue = Math.floor(startValue + (targetValue - startValue) * easeOutQuart);
        
        element.textContent = currentValue.toLocaleString();
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            element.textContent = targetValue.toLocaleString();
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function handleResize() {
    // Resize charts on window resize
    if (window.activityChart) window.activityChart.resize();
    if (window.distributionChart) window.distributionChart.resize();
    if (window.toxicityChart) window.toxicityChart.resize();
    if (window.ratingChart) window.ratingChart.resize();
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Global functions for HTML onclick handlers
window.navigateTo = navigateTo;
window.showDetails = showDetails;