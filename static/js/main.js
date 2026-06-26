import { goToLogin, handleAccess, showLoader, hideLoader, showAiOutput, setupScrollReveal } from './ui.js';
import { runSketch, runSegmentation } from './cv_logic.js';
import { geminiCall, generateGhibliStyle } from './api_client.js';

// Application State
const appState = {
    currentImgBase64: null,
    bodyPixNet: null,
    rawImg: new Image(),
    canvas: null,
    ctx: null
};

// Global flags for 3rd party scripts
window.cvReady = false;

document.addEventListener('DOMContentLoaded', () => {
    // UI Event Binding
    document.getElementById('btnLaunch').addEventListener('click', goToLogin);
    document.getElementById('authForm').addEventListener('submit', (e) => handleAccess(e, initTF));
    document.getElementById('btnCloseAi').addEventListener('click', (e) => {
        e.currentTarget.parentElement.parentElement.classList.add('hidden');
    });

    setupScrollReveal();

    // Init Engine Waiters
    window.Module = {
        onRuntimeInitialized: () => {
            window.cvReady = true;
            console.log("OpenCV Runtime Ready");
        }
    };
    
    const checkCV = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            window.cvReady = true;
            clearInterval(checkCV);
            console.log("OpenCV Object Detected");
        }
    }, 500);

    // Canvas & File Handling
    appState.canvas = document.getElementById('outputCanvas');
    appState.ctx = appState.canvas.getContext('2d', { willReadFrequently: true });
    
    const imgInput = document.getElementById('imageInput');
    
    document.getElementById('triggerUpload').onclick = () => imgInput.click();

    imgInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (ev) => {
            appState.currentImgBase64 = ev.target.result;
            appState.rawImg.src = ev.target.result;

            appState.rawImg.onload = () => {
                document.getElementById('empty').style.display = 'none';
                appState.canvas.style.display = 'block';
                ['btnSketch', 'btnCutout', 'btnGhibli', 'btnCaption', 'btnSuggest']
                    .forEach(id => document.getElementById(id).disabled = false);
                document.getElementById('promptInput').classList.remove('hidden');
                document.getElementById('btnExport').style.display = 'block';
                
                appState.canvas.width = appState.rawImg.width;
                appState.canvas.height = appState.rawImg.height;
                appState.ctx.drawImage(appState.rawImg, 0, 0);
            };
        };
        reader.readAsDataURL(file);
    });

    // Toolbar Buttons
    document.getElementById('btnSketch').onclick = () => runSketch(appState.canvas);
    
    document.getElementById('btnCutout').onclick = () => runSegmentation(
        appState.rawImg, 
        appState.canvas, 
        appState.ctx, 
        appState.bodyPixNet
    );

    document.getElementById('btnGhibli').onclick = () => {
        const customPrompt = document.getElementById('promptInput').value;
        generateGhibliStyle(
            appState.currentImgBase64, 
            customPrompt,
            appState.canvas, 
            appState.ctx
        );
    };

    document.getElementById('btnCaption').onclick = async () => {
        showLoader("SYNTHESIZING ✨");
        const res = await geminiCall("Write a single poetic, high-end caption for this artwork. Keep it under 20 words.", appState.currentImgBase64);
        showAiOutput(res);
        hideLoader();
    };

    document.getElementById('btnSuggest').onclick = async () => {
        showLoader("CONSULTING AI ✨");
        const res = await geminiCall("As a professional art critic, suggest 3 concise improvements for this piece's composition and lighting.", appState.currentImgBase64);
        showAiOutput(res);
        hideLoader();
    };

    document.getElementById('btnExport').onclick = () => {
        const link = document.createElement('a');
        link.download = "studio-ar-export.png";
        link.href = appState.canvas.toDataURL();
        link.click();
    };
});

async function initTF() {
    try {
        if (window.tf) {
            await tf.ready();
            if (typeof bodyPix !== 'undefined') {
                appState.bodyPixNet = await bodyPix.load();
                console.log("BodyPix Loaded Successfully");
            } else {
                setTimeout(initTF, 1000);
            }
        } else {
            setTimeout(initTF, 500);
        }
    } catch(e) {
        console.error("TF Error:", e);
    }
}
