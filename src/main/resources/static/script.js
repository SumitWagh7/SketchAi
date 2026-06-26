const apiKey = ""; 
let currentImgBase64 = null;
let bodyPixNet = null;
let rawImg = new Image();
let cvReady = false;

// ✅ Robust OpenCV initialization
window.Module = {
    onRuntimeInitialized: () => {
        cvReady = true;
        console.log("OpenCV Runtime Ready");
    }
};

const checkCV = setInterval(() => {
    if (typeof cv !== 'undefined' && cv.Mat) {
        cvReady = true;
        clearInterval(checkCV);
    }
}, 500);

// --- NAVIGATION ---
function goToLogin() {
    const landing = document.getElementById('landingView');
    landing.style.opacity = '0';
    landing.style.transform = 'scale(0.98)';
    setTimeout(() => {
        landing.style.display = 'none';
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

function handleAccess(e) {
    e.preventDefault();
    const auth = document.getElementById('authContainer');
    auth.style.opacity = '0';
    auth.style.transform = 'scale(0.95)';
    setTimeout(() => {
        auth.style.display = 'none';
        document.getElementById('appWorkspace').style.display = 'grid';
        initTF();
    }, 600);
}

// --- AI & PROCESSING ---
async function initTF() {
    try {
        if (window.tf) {
            await tf.ready();
            bodyPixNet = await bodyPix.load();
        }
    } catch(e){ console.error(e); }
}

const imgInput = document.getElementById('imageInput');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

document.getElementById('triggerUpload').onclick = () => imgInput.click();

imgInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        currentImgBase64 = ev.target.result;
        rawImg.src = ev.target.result;
        rawImg.onload = () => {
            document.getElementById('empty').style.display = 'none';
            canvas.style.display = 'block';
            ['btnSketch', 'btnCutout', 'btnCaption', 'btnSuggest', 'btnGhibli'].forEach(id => document.getElementById(id).disabled = false);
            document.getElementById('btnExport').style.display = 'block';
            
            // Limit canvas size for performance/stability
            const maxDim = 1200;
            let w = rawImg.width;
            let h = rawImg.height;
            if (w > maxDim || h > maxDim) {
                const scale = Math.min(maxDim/w, maxDim/h);
                w *= scale;
                h *= scale;
            }
            
            canvas.width = w;
            canvas.height = h;
            ctx.drawImage(rawImg, 0, 0, w, h);
        };
    };
    reader.readAsDataURL(file);
});

// --- GHIBLI ART FILTER (LEGACY OPENCV - DISABLED) ---
// The following handler was disabled as it represents the legacy local OpenCV implementation.
// The active Ghibli workflow routes requests to Spring Boot -> Python -> SDXL Turbo.
/*
document.getElementById('btnGhibli').onclick = () => {
    if (!cvReady) return alert("OpenCV is still loading. Please wait a moment.");
    showLoader("PAINTING GHIBLI STYLE");
    
    // Small delay to ensure loader renders
    setTimeout(() => {
        try {
            // 1. Read source
            let src = cv.imread(canvas);
            let dst = new cv.Mat();
            
            // 2. Format conversion: Ghibli look needs smoothing (Bilateral)
            // cv.imread returns RGBA, Bilateral expects RGB or Gray
            let rgb = new cv.Mat();
            cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
            
            // Apply Bilateral Filter - smoothing while preserving edges
            // Arguments: src, dst, diameter, sigmaColor, sigmaSpace
            cv.bilateralFilter(rgb, dst, 12, 80, 80);
            
            // 3. Color Grading: Warmth and Saturation
            let hsv = new cv.Mat();
            cv.cvtColor(dst, hsv, cv.COLOR_RGB2HSV);
            let channels = new cv.MatVector();
            cv.split(hsv, channels);
            
            // Boost Saturation (Channel index 1)
            let sat = channels.get(1);
            sat.convertTo(sat, -1, 1.3, 15); 
            
            // Boost Brightness (Channel index 2)
            let val = channels.get(2);
            val.convertTo(val, -1, 1.1, 5);
            
            // Reassemble
            cv.merge(channels, hsv);
            cv.cvtColor(hsv, dst, cv.COLOR_HSV2RGB);
            
            // 4. Subtle Edge Overlay for that hand-drawn look
            let edges = new cv.Mat();
            cv.cvtColor(rgb, edges, cv.COLOR_RGB2GRAY);
            cv.adaptiveThreshold(edges, edges, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 7);
            cv.cvtColor(edges, edges, cv.COLOR_GRAY2RGB);
            
            // Blend edges slightly
            cv.multiply(dst, edges, dst, 1/255);
            
            // Show Output
            cv.imshow('outputCanvas', dst);
            
            // Cleanup (Crucial for OpenCV in Browser)
            [src, dst, rgb, hsv, sat, val, edges].forEach(m => { if(m) m.delete(); });
            channels.delete();
        } catch (e) { 
            console.error("Ghibli Processing Error:", e);
            alert("Ghibli processing failed. Your browser might be out of memory for this image size.");
        }
        hideLoader();
    }, 50);
};
*/

document.getElementById('btnSketch').onclick = () => {
    if (!cvReady) return alert("OpenCV loading...");
    showLoader("EXTRACTING LINES");
    setTimeout(() => {
        try {
            let src = cv.imread(canvas);
            let gray = new cv.Mat();
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            let inv = new cv.Mat();
            cv.bitwise_not(gray, inv);
            let blur = new cv.Mat();
            cv.GaussianBlur(inv, blur, new cv.Size(21, 21), 0);
            let invBlur = new cv.Mat();
            cv.bitwise_not(blur, invBlur);
            let sketch = new cv.Mat();
            cv.divide(gray, invBlur, sketch, 256.0);
            cv.imshow('outputCanvas', sketch);
            [src, gray, inv, blur, invBlur, sketch].forEach(m => m.delete());
        } catch (e) { console.error(e); }
        hideLoader();
    }, 50);
};

document.getElementById('btnCutout').onclick = async () => {
    if (!bodyPixNet) return;
    showLoader("AI SEGMENTATION");
    try {
        const seg = await bodyPixNet.segmentPerson(canvas);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < seg.data.length; i++) { 
            if (seg.data[i] === 0) data.data[i * 4 + 3] = 0; 
        }
        ctx.putImageData(data, 0, 0);
    } catch(e) { console.error(e); }
    hideLoader();
};

async function geminiCall(prompt) {
    if (!apiKey) return "⚠️ Add API key to script.";
    try {
        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=${apiKey}`;
        const payload = { contents: [{ parts: [{ text: prompt }, { inlineData: { mimeType: "image/png", data: currentImgBase64.split(',')[1] } }] }] };
        const resp = await fetch(endpoint, { method: 'POST', body: JSON.stringify(payload) });
        const data = await resp.json();
        return data.candidates?.[0]?.content?.parts?.[0]?.text || "No response";
    } catch(e) { return "AI Error"; }
}

document.getElementById('btnCaption').onclick = async () => {
    showLoader("SYNTHESIZING ✨");
    const res = await geminiCall("Write a poetic caption for this image.");
    showAiOutput(res);
    hideLoader();
};

document.getElementById('btnSuggest').onclick = async () => {
    showLoader("ART DIRECTION ✨");
    const res = await geminiCall("Suggest 3 ways to improve this composition.");
    showAiOutput(res);
    hideLoader();
};

document.getElementById('btnExport').onclick = () => {
    const link = document.createElement('a');
    link.download = "sketch.png";
    link.href = canvas.toDataURL();
    link.click();
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => { if (entry.isIntersecting) entry.target.classList.add('active'); });
}, { threshold: 0.1 });
document.querySelectorAll('.reveal-on-scroll').forEach(el => observer.observe(el));

function showLoader(t) { document.getElementById('mainLoader').style.display = 'flex'; document.getElementById('loaderText').innerText = t; }
function hideLoader() { document.getElementById('mainLoader').style.display = 'none'; }
function showAiOutput(t) { document.getElementById('aiOutput').style.display = 'block'; document.getElementById('aiContent').innerText = t; }
