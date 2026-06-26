// Handles UI State and Animations

export function goToLogin() {
    document.getElementById('landingView').style.opacity = '0';
    document.getElementById('landingView').style.transform = 'scale(0.98)';
    setTimeout(() => {
        document.getElementById('landingView').style.display = 'none';
        document.body.classList.add('studio-active');
        document.getElementById('studioBg').style.display = 'block';
        const auth = document.getElementById('authContainer');
        auth.style.display = 'block';
        setTimeout(() => {
            auth.style.opacity = '1';
            auth.style.transform = 'translateY(0)';
        }, 50);
    }, 600);
}

export function handleAccess(e, successCallback) {
    e.preventDefault();
    document.getElementById('authContainer').style.opacity = '0';
    setTimeout(() => {
        document.getElementById('authContainer').style.display = 'none';
        document.getElementById('appWorkspace').style.display = 'grid';
        if(successCallback) successCallback();
    }, 600);
}

export function showLoader(t) {
    document.getElementById('mainLoader').style.display = 'flex';
    document.getElementById('loaderText').innerText = t;
}

export function hideLoader() {
    document.getElementById('mainLoader').style.display = 'none';
}

export function showAiOutput(t) {
    document.getElementById('aiOutput').classList.remove('hidden');
    document.getElementById('aiContent').innerText = t;
}

export function setupScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => { 
            if (entry.isIntersecting) entry.target.classList.add('active'); 
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));
}
