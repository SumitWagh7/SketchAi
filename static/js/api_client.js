import { showLoader, hideLoader } from './ui.js';

export async function geminiCall(prompt, currentImgBase64) {
    if (!currentImgBase64) return "Error: No image found.";
    const base64Data = currentImgBase64.split(',')[1];
    if (!base64Data) return "Error: Image not processed correctly.";

    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: prompt, image: base64Data })
        });
        
        if (!response.ok) throw new Error("HTTP error " + response.status);
        const data = await response.json();
        return data.text || data.error || "No response received via Python API.";
    } catch (error) {
        console.error("Gemini Error:", error);
        return "Failed to connect to Python Backend API.";
    }
}

export async function generateGhibliStyle(currentImgBase64, prompt, canvas, ctx, retries = 0) {
    if (!currentImgBase64) {
        alert("Please upload an image first!");
        return;
    }
    
    if (retries === 0) showLoader("PAINTING GHIBLI SCENE...");
    const base64Data = currentImgBase64.split(',')[1];

    try {
        const response = await fetch('/api/ghibli', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64Data, prompt: prompt })
        });

        if (!response.ok) throw new Error('Backend API Error');
        const data = await response.json();
        
        if (data.error) throw new Error(data.error);

        const finalUrl = `data:image/png;base64,${data.image}`;
        const ghibliImg = new Image();
        
        ghibliImg.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(ghibliImg, 0, 0, canvas.width, canvas.height);
            hideLoader();
        };
        ghibliImg.src = finalUrl;
        
    } catch (err) {
        if (retries < 5) {
            const delay = Math.pow(2, retries) * 1000;
            await new Promise(res => setTimeout(res, delay));
            return generateGhibliStyle(currentImgBase64, prompt, canvas, ctx, retries + 1);
        }
        console.error(err);
        alert("Transformation failed. Please try again.");
        hideLoader();
    }
}
