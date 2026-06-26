document.addEventListener("DOMContentLoaded", () => {
    const card = document.querySelector('.glass-card');
    
    // Add 3D rotation based on mouse movement
    document.addEventListener('mousemove', (e) => {
        // Calculate the mouse position relative to the center of the viewport
        const xAxis = (window.innerWidth / 2 - e.pageX) / 40;
        const yAxis = (window.innerHeight / 2 - e.pageY) / 40;
        
        // Apply rotation to the card
        card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
    });

    // Reset rotation when the mouse leaves the document
    document.addEventListener('mouseleave', () => {
        card.style.transform = `rotateY(0deg) rotateX(0deg)`;
        card.style.transition = 'transform 0.5s ease-out'; // Add transition for smooth reset
    });

    // Remove transition when moving to keep it snappy
    document.addEventListener('mouseenter', () => {
        card.style.transition = 'none';
    });
});
