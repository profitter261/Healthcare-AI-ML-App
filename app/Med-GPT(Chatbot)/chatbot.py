from flask import Flask, render_template_string

app = Flask(__name__)

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare AI</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Poppins:wght@400;500;700&display=swap');

:root {
    --glow-color: #1d9bf0;
    --secondary-glow: #00ffd9;
    --shadow-light: 0 0 10px rgba(29, 155, 240, 0.6), 0 0 20px rgba(29, 155, 240, 0.4);
    --text-color: #fff;
    --input-bg-dark: #1f1f1f;
    --input-bg-light: #f0f8ff; 
    --input-text-dark: #0a0a0a;
    --error-color: #ee5253;
}

.light-mode {
    background: #f4f7f6 !important;
    color: #333 !important;
    --glow-color: #007bff;
    --secondary-glow: #00a68c;
    --shadow-light: 0 5px 15px rgba(0,0,0,0.1);
    --text-color: #333;
    --input-bg-dark: #fff;
    --input-bg-light: #fff; 
    --input-text-dark: #333;
    --error-color: #c0392b;
}

body {
    margin: 0;
    font-family: 'Poppins', sans-serif;
    background: radial-gradient(circle at center, #0b0c10, #001f3f);
    color: var(--text-color);
    transition: background 0.5s, color 0.5s;
    overflow-x: hidden;
    display: flex;
}

body.light-mode {
    background: #f4f7f6 !important;
    color: #333 !important;
}

body.dark-mode {
    background: radial-gradient(circle at center, #0b0c10, #001f3f);
    color: var(--text-color);
}

/* Navigation panels */
.nav-panel {
    position: fixed;
    top: 0;
    width: 18%;
    min-width: 180px;
    max-width: 250px;
    height: 100vh;
    background: rgba(255,255,255,0.08); 
    backdrop-filter: blur(12px);
    box-shadow: 0 0 20px rgba(0,0,0,0.6);
    transition: transform 0.5s ease;
    z-index: 10;
    padding: 2rem 1rem;
    overflow-y: auto;
}

.light-mode .nav-panel {
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.left-nav { left: 0; border-radius: 0 20px 20px 0; transform: translateX(0); }
.right-nav { right: 0; border-radius: 20px 0 0 20px; transform: translateX(0); }
.nav-hidden-left { transform: translateX(-100%); }
.nav-hidden-right { transform: translateX(100%); }

.nav-panel h2 { 
    color: var(--glow-color); 
    text-align: center; 
    margin-bottom: 2rem; 
    font-size: 1.5rem; 
    text-shadow: 0 0 8px var(--glow-color); 
}

.light-mode .nav-panel h2 { text-shadow: none; }

.nav-panel ul { list-style: none; padding: 0; }
.nav-panel li { margin: 1.5rem 0; text-align: center; font-weight: 500; cursor: pointer; transition: all 0.3s ease; }

.nav-panel li a { 
    color: var(--secondary-glow); 
    text-decoration: none; 
    display: flex; 
    align-items: center;
    justify-content: center;
    padding: 10px 0;
    text-shadow: 0 0 5px rgba(255,255,255,0.2);
}

.light-mode .nav-panel li a { text-shadow: none; }

.nav-panel li a:hover { 
    color: var(--glow-color); 
    text-shadow: 0 0 10px var(--glow-color); 
    transform: scale(1.05); 
}

.light-mode .nav-panel li a:hover { text-shadow: none; color: var(--glow-color); }

/* Toggle Buttons */
.toggle-btn {
    position: fixed;
    top: 50%;
    transform: translateY(-50%);
    background: rgba(0,0,0,0.7);
    border: 2px solid var(--glow-color);
    box-shadow: var(--shadow-light);
    color: var(--glow-color);
    font-size: 1.2rem;
    padding: 0.8rem 0.6rem;
    cursor: pointer;
    z-index: 30;
    transition: all 0.5s ease;
}

.light-mode .toggle-btn {
    background: #fff;
    border: 2px solid var(--glow-color);
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.toggle-btn:hover { 
    background: rgba(0,0,0,1); 
    box-shadow: 0 0 15px var(--glow-color), 0 0 30px var(--glow-color);
}

.light-mode .toggle-btn:hover { background: #eee; box-shadow: 0 2px 10px var(--glow-color);}

.left-toggle { left: 0; border-radius: 0 10px 10px 0; }
.right-toggle { right: 0; border-radius: 10px 0 0 10px; }

/* Theme toggle */
.theme-toggle {
    position: absolute;
    top: 1rem;
    right: 1rem;
    z-index: 40;
}
.theme-btn {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 1.8rem;
  color: var(--secondary-glow);
  transition: transform 0.3s ease, color 0.3s ease;
}

.theme-btn:hover {
  transform: rotate(20deg);
}

body.light-mode .theme-btn {
  color: var(--text-color);
}

/* Sidebar */
.sidebar {
    width: 200px;
    background: #16213e;
    padding: 20px 0;
    border-right: 1px solid #0f3460;
    margin-top: 40px;
}

.light-mode .sidebar {
    background: #f0f0f0;
    border-right: 1px solid #ddd;
}

.sidebar-header {
    padding: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #fff;
    font-weight: bold;
    margin-bottom: 20px;
    justify-content: flex-start;
}

.light-mode .sidebar-header { color: #333; }

.sidebar-nav { list-style: none; }
.sidebar-nav li {
    padding: 12px 20px;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 10px;
    color: #a0a0a0;
    justify-content: flex-start;
}
.light-mode .sidebar-nav li { color: #666; }
.sidebar-nav li:hover { background: #0f3460; color: #fff; padding-left: 30px; }
.light-mode .sidebar-nav li:hover { background: #e0e0e0; color: #333; }
.sidebar-nav li.active { background: #0f3460; color: #00ffd9; border-left: 4px solid #00ffd9; padding-left: 26px; }
.light-mode .sidebar-nav li.active { background: #d0d0d0; color: var(--secondary-glow); }

.page-message { text-align: center; padding: 20px; background: #0d1b2a; border-radius: 8px; color: #00ffd9; margin: 40px 20px 0; }
.light-mode .page-message { background: #e9f5ff; color: var(--secondary-glow); }

/* Main content */
.main-content { flex: 1; padding: 40px; display: flex; flex-direction: column; }
.page { display: none; }
.page.active { display: flex; flex-direction: column; gap: 20px; }
.page-title { font-size: 2rem; color: #00ffd9; margin-bottom: 10px; text-shadow: 0 0 10px rgba(0, 255, 217, 0.3); }
.light-mode .page-title { text-shadow: none; color: var(--secondary-glow); }
.page-subtitle { color: #a0a0a0; margin-bottom: 30px; }
.light-mode .page-subtitle { color: #666; }

/* Form */
.form-group { display: flex; flex-direction: column; gap: 8px; margin-bottom: 20px; max-width: 400px; }
.form-group label { color: #00ffd9; font-weight: 500; }
.light-mode .form-group label { color: var(--secondary-glow); }
.form-group input, .form-group textarea {
    padding: 12px;
    background: #0f3460;
    border: 1px solid #00ffd9;
    border-radius: 6px;
    color: #e0e0e0;
    font-family: 'Poppins', sans-serif;
    font-size: 0.95rem;
}
.light-mode .form-group input, .light-mode .form-group textarea { background: #fff; border: 1px solid var(--secondary-glow); color: #333; }
.form-group input:focus, .form-group textarea:focus { outline: none; border-color: #1d9bf0; box-shadow: 0 0 10px rgba(29,155,240,0.3); }
.form-group textarea { resize: vertical; min-height: 100px; }

.btn { background: linear-gradient(135deg, #1d9bf0 0%, #00ffd9 100%); color: white; border: none; padding: 12px 30px; border-radius: 6px; cursor: pointer; font-weight: 600; width: fit-content; transition: all 0.3s ease; box-shadow: 0 0 15px rgba(29,155,240,0.3); }
.btn:hover { transform: translateY(-2px); box-shadow: 0 0 25px rgba(29,155,240,0.5); }

/* FAQ */
.faq-container { width: 100%; max-width: 600px; background: #16213e; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; max-height: 500px; }
.light-mode .faq-container { background: #f9f9f9; border: 1px solid #ddd; }
.faq-body { padding: 20px; height: 400px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 15px; }
.faq-body::-webkit-scrollbar { width: 6px; }
.faq-body::-webkit-scrollbar-thumb { background: #00ffd9; border-radius: 10px; }
.message { padding: 12px 16px; border-radius: 10px; max-width: 85%; word-wrap: break-word; font-size: 0.95rem; line-height: 1.4; }
.user-message { background: linear-gradient(135deg, #1d9bf0 0%, #00ffd9 100%); color: white; align-self: flex-end; border-radius: 10px 2px 10px 10px; }
.bot-message { background: #0f3460; color: #e0e0e0; align-self: flex-start; border-radius: 2px 10px 10px 10px; max-width: 95%; border-left: 3px solid #00ffd9; }
.light-mode .bot-message { background: #e9f5ff; color: #333; border-left: 3px solid var(--secondary-glow); }
.options-container { display: flex; flex-direction: column; gap: 10px; margin-top: 10px; }
.option-btn { background: linear-gradient(135deg, #1d9bf0 0%, #00ffd9 100%); color: white; border: none; padding: 10px 14px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; font-weight: 500; transition: all 0.3s ease; text-align: left; }
.option-btn:hover { transform: translateY(-2px); }
.back-btn { background: linear-gradient(135deg, #666666 0%, #444444 100%); }

.med-gpt-blank { display: flex; justify-content: center; align-items: center; height: 100%; min-height: 400px; }
</style>
</head>
<body class="dark-mode">

<div class="theme-toggle">
  <button id="themeToggle" class="theme-btn">
    <i class="bi bi-moon-fill"></i>
  </button>
</div>


<div class="nav-panel left-nav" id="leftNav">
    <h2>Navigation</h2>
    <ul>
        <li><a href="/"><i class="bi bi-house-door"></i> Home</a></li>
        <li><a href="/Disease"><i class="bi bi-heart-pulse"></i> Disease Prediction</a></li>
        <li><a href="/LOS"><i class="bi bi-graph-up"></i> LOS Prediction</a></li>
        <li><a href="/Cluster"><i class="bi bi-people"></i> Patient Cohorts</a></li>
        <li><a href="/chat"><i class="bi bi-question-circle"></i> Help</a></li>
    </ul>
</div>

<div class="nav-panel right-nav" id="rightNav">
    <h2>AI Tools</h2>
    <ul>
        <li><a href="/Patient"><i class="bi bi-lightbulb"></i> Patient Risk Assessment</a></li>
        <li><a href="/Summarizer"><i class="bi bi-file-earmark-text"></i> Notes Summarizer</a></li>
        <li><a href="/Image-Diagnostics"><i class="bi bi-image"></i> Image Diagnostics</a></li>
        <li><a href="/Senti"><i class="bi bi-chat-dots"></i> Feedback Analysis</a></li>
        <li><a href="/admin"><i class="bi bi-table"></i> Admin Dashboard</a></li>
    </ul>
</div>

<button class="toggle-btn left-toggle" id="leftToggle">☰</button>
<button class="toggle-btn right-toggle" id="rightToggle">☰</button>

<!-- Sidebar -->
<div class="sidebar">
  <div class="sidebar-header">
    <span>Menu</span>
  </div>
  <ul class="sidebar-nav">
    <li class="nav-item active" data-page="faq">
      <span>FAQ</span>
    </li>
    <li class="nav-item" data-page="med-gpt">
      <span>Med-GPT</span>
    </li>
    <li class="nav-item" data-page="contact">
      <span>Contact</span>
    </li>
  </ul>
  <div class="page-message">Select a page above.</div>
</div>

<!-- Main Content -->
<div class="main-content">
  <!-- FAQ Page -->
  <div class="page active" id="faq-page">
    <h1 class="page-title">Healthcare AI FAQ Bot</h1>
    <p class="page-subtitle">Get answers to common questions</p>
    <div class="faq-container">
      <div class="faq-body" id="chat-body"></div>
    </div>
  </div>

  <!-- Med-GPT Page -->
  <div class="page" id="med-gpt-page">
    <div class="med-gpt-blank"></div>
  </div>

  <!-- Contact Page -->
  <div class="page" id="contact-page">
    <h1 class="page-title">Contact</h1>
    <p class="page-subtitle">Get in touch with us</p>
    <div class="form-group">
      <label>Email</label>
      <input type="email" placeholder="Enter your email">
    </div>
    <div class="form-group">
      <label>Message</label>
      <textarea placeholder="Enter your message..."></textarea>
    </div>
    <button class="btn">Submit</button>
  </div>
</div>

<script>
const body = document.body;
const leftNav = document.getElementById('leftNav');
const rightNav = document.getElementById('rightNav');
const leftToggle = document.getElementById('leftToggle');
const rightToggle = document.getElementById('rightToggle');
const themeToggle = document.getElementById('themeToggle');
const navItems = document.querySelectorAll('.nav-item');
const pages = document.querySelectorAll('.page');
const chatBody = document.getElementById('chat-body');
const themeToggleBtn = document.getElementById('themeToggle');
const icon = themeToggleBtn.querySelector('i');
const savedTheme = localStorage.getItem('theme');

// Apply saved theme on load
if (savedTheme === 'light') {
  body.classList.add('light-mode');
  icon.classList.replace('bi-moon-fill', 'bi-sun-fill');
} else {
  body.classList.add('dark-mode');
  icon.classList.replace('bi-sun-fill', 'bi-moon-fill');
}

// Toggle theme on click
themeToggleBtn.addEventListener('click', () => {
  if (body.classList.contains('dark-mode')) {
    body.classList.remove('dark-mode');
    body.classList.add('light-mode');
    icon.classList.replace('bi-moon-fill', 'bi-sun-fill');
    localStorage.setItem('theme', 'light');
  } else {
    body.classList.remove('light-mode');
    body.classList.add('dark-mode');
    icon.classList.replace('bi-sun-fill', 'bi-moon-fill');
    localStorage.setItem('theme', 'dark');
  }
});

// Navigation toggles
leftToggle.addEventListener('click', () => {
    leftNav.classList.toggle('nav-hidden-left');
});
rightToggle.addEventListener('click', () => {
    rightNav.classList.toggle('nav-hidden-right');
});

// Page navigation
navItems.forEach(item => {
  item.addEventListener('click', () => {
    navItems.forEach(nav => nav.classList.remove('active'));
    pages.forEach(page => page.classList.remove('active'));
    
    item.classList.add('active');
    const pageId = item.dataset.page + '-page';
    document.getElementById(pageId).classList.add('active');
    
    if (item.dataset.page === 'faq') showMainMenu();
  });
});

// FAQ functionality
const faqStructure = {
  greeting: "Hello! Welcome to Healthcare AI FAQ Bot. What would you like to know about?",
  mainCategories: [
    { text: "Healthcare AI in General", key: "general" },
    { text: "Healthcare AI App", key: "app" },
    { text: "App Features", key: "features" }
  ],
  general: { title: "Healthcare AI in General", questions: [
    { text: "What is Healthcare AI?", key: "what_is_ai" },
    { text: "Why is Healthcare AI important?", key: "why_important" },
    { text: "How is it different from regular apps?", key: "difference" }
  ]},
  app: { title: "Healthcare AI App", questions: [
    { text: "How does your app work?", key: "how_app_works" },
    { text: "Is it safe and secure?", key: "safety" },
    { text: "Who can use this app?", key: "who_can_use" }
  ]},
  features: { title: "App Features", questions: [
    { text: "Symptom Checker", key: "symptom_checker" },
    { text: "Health Tracking", key: "health_tracking" },
    { text: "AI Suggestions", key: "ai_suggestions" },
    { text: "Health Tips", key: "health_tips" },
    { text: "Real-time Monitoring", key: "monitoring" }
  ]},
  answers: {
    what_is_ai: "Healthcare AI refers to AI applications designed to improve healthcare services, diagnostics, and patient care.",
    why_important: "Healthcare AI is important because it makes healthcare more accessible and personalized.",
    difference: "Healthcare AI uses machine learning to understand your unique health patterns and provides tailored recommendations.",
    how_app_works: "Our app collects your health data and provides personalized insights using AI algorithms.",
    safety: "Yes, all data is encrypted and complies with healthcare regulations.",
    who_can_use: "Anyone looking to improve their healthcare monitoring can use it.",
    symptom_checker: "Input symptoms and get preliminary analysis.",
    health_tracking: "Track vital signs, medications, diet, exercise, and sleep.",
    ai_suggestions: "AI provides personalized recommendations based on your health data.",
    health_tips: "Receive daily personalized health tips.",
    monitoring: "Get real-time alerts and notifications based on your health data."
  }
};

function appendMessage(message, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.classList.add("message", sender === "user" ? "user-message" : "bot-message");
  messageDiv.innerText = message;
  chatBody.appendChild(messageDiv);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function appendBotMessage(message) { appendMessage(message, "bot"); }

function showOptions(options, showBack = false) {
  const optionsContainer = document.createElement("div");
  optionsContainer.classList.add("options-container");
  options.forEach(option => {
    const btn = document.createElement("button");
    btn.classList.add("option-btn");
    btn.innerText = option.text;
    btn.addEventListener("click", () => handleMainCategoryClick(option));
    optionsContainer.appendChild(btn);
  });
  if (showBack) {
    const backBtn = document.createElement("button");
    backBtn.classList.add("option-btn", "back-btn");
    backBtn.innerText = "Back to Main Menu";
    backBtn.addEventListener("click", () => showMainMenu());
    optionsContainer.appendChild(backBtn);
  }
  chatBody.appendChild(optionsContainer);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function showMainMenu() {
  chatBody.innerHTML = "";
  appendBotMessage(faqStructure.greeting);
  showOptions(faqStructure.mainCategories);
}

function handleMainCategoryClick(option) {
  appendMessage(option.text, "user");
  const category = faqStructure[option.key];
  if (category.questions) {
    appendBotMessage(category.title);
    showOptions(category.questions, true);
  } else if (faqStructure.answers[option.key]) {
    appendBotMessage(faqStructure.answers[option.key]);
    const backBtn = document.createElement("div");
    backBtn.className = "options-container";
    const btn = document.createElement("button");
    btn.classList.add("option-btn", "back-btn");
    btn.innerText = "Back to Main Menu";
    btn.addEventListener("click", () => showMainMenu());
    backBtn.appendChild(btn);
    chatBody.appendChild(backBtn);
    chatBody.scrollTop = chatBody.scrollHeight;
  }
}

// Initialize
initializeTheme();
showMainMenu();
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html_content)

if __name__ == "__main__":
    app.run(debug=True)
