import { showLoader, hideLoader } from './ui.js';

export function runSketch(canvas) {
    if (typeof cv === 'undefined' || !window.cvReady) {
        alert("Processing engine (OpenCV) is still loading...");
        return;
    }
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
        } catch (e) {
            console.error(e); 
        }
        hideLoader();
    }, 100);
}

export async function runSegmentation(rawImg, canvas, ctx, bodyPixNet) {
    if (!bodyPixNet) {
        alert("AI model is still loading.");
        return;
    }
    showLoader("AI SEGMENTATION");
    try {
        const seg = await bodyPixNet.segmentPerson(rawImg);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < seg.data.length; i++) {
            if (seg.data[i] === 0) data.data[i * 4 + 3] = 0;
        }
        ctx.putImageData(data, 0, 0);
    } catch(e) { console.error(e); }
    hideLoader();
}

export function runGhibliStyle(canvas) {
    if (typeof cv === 'undefined' || !window.cvReady) {
        alert("Processing engine (OpenCV) is still loading...");
        return;
    }
    showLoader("PAINTING GHIBLI STYLE");
    
    setTimeout(() => {
        try {
            let src = cv.imread(canvas);
            let dst = new cv.Mat();
            
            let rgb = new cv.Mat();
            cv.cvtColor(src, rgb, cv.COLOR_RGBA2RGB);
            
            cv.bilateralFilter(rgb, dst, 12, 80, 80);
            
            let hsv = new cv.Mat();
            cv.cvtColor(dst, hsv, cv.COLOR_RGB2HSV);
            let channels = new cv.MatVector();
            cv.split(hsv, channels);
            
            let sat = channels.get(1);
            sat.convertTo(sat, -1, 1.3, 15); 
            
            let val = channels.get(2);
            val.convertTo(val, -1, 1.1, 5);
            
            cv.merge(channels, hsv);
            cv.cvtColor(hsv, dst, cv.COLOR_HSV2RGB);
            
            let edges = new cv.Mat();
            cv.cvtColor(rgb, edges, cv.COLOR_RGB2GRAY);
            cv.adaptiveThreshold(edges, edges, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 7);
            cv.cvtColor(edges, edges, cv.COLOR_GRAY2RGB);
            
            cv.multiply(dst, edges, dst, 1/255);
            
            cv.imshow('outputCanvas', dst);
            
            [src, dst, rgb, hsv, sat, val, edges].forEach(m => { if(m) m.delete(); });
            channels.delete();
        } catch (e) { 
            console.error("Ghibli Processing Error:", e);
            alert("Ghibli processing failed. Your browser might be out of memory for this image size.");
        }
        hideLoader();
    }, 50);
}
