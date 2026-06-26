package com.project.sketchai.service;

import com.project.sketchai.dto.AiRequestDto;
import com.project.sketchai.model.AiRequest;
import com.project.sketchai.model.User;
import com.project.sketchai.repository.AiRequestRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class AiService {

    private final AiRequestRepository aiRequestRepository;
    private final UserService userService;
    private final RestTemplate restTemplate;

    @Value("${python.ai.engine.url:http://localhost:8000}")
    private String pythonEngineUrl;

    public AiService(AiRequestRepository aiRequestRepository, UserService userService) {
        this.aiRequestRepository = aiRequestRepository;
        this.userService = userService;
        
        org.springframework.http.client.SimpleClientHttpRequestFactory factory = new org.springframework.http.client.SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);  // 5s connect timeout
        factory.setReadTimeout(120000);   // 120s read timeout (long AI tasks)
        this.restTemplate = new RestTemplate(factory);
    }

    private <T> T callWithRetry(String url, HttpEntity<?> entity, Class<T> responseType, String actionType) {
        int maxAttempts = 3;
        int delayMs = 1000;
        Exception lastException = null;
        
        for (int attempt = 1; attempt <= maxAttempts; attempt++) {
            try {
                System.out.println("AI Service: Sending " + actionType + " request to Python engine at " + url + " (attempt " + attempt + "/" + maxAttempts + ")...");
                ResponseEntity<T> response = restTemplate.postForEntity(url, entity, responseType);
                return response.getBody();
            } catch (Exception e) {
                lastException = e;
                System.err.println("AI Service: Attempt " + attempt + " failed. Error: " + e.getMessage());
                if (attempt < maxAttempts) {
                    try {
                        Thread.sleep(delayMs * attempt); // backoff: 1s, then 2s
                    } catch (InterruptedException ie) {
                        Thread.currentThread().interrupt();
                        throw new RuntimeException("Retry interrupted", ie);
                    }
                }
            }
        }
        
        throw new RuntimeException("Python AI Engine connection failed after " + maxAttempts + " attempts: " + lastException.getMessage(), lastException);
    }

    public String generateCaptionOrSuggestion(AiRequestDto request, String actionType) {
        User user = userService.getCurrentUser();
        String endpoint = actionType.equals("STRATEGY") ? "/api/python/edit/strategy" : "/api/python/caption";
        String url = pythonEngineUrl + endpoint;
        
        Map<String, Object> payload = new HashMap<>();
        payload.put("prompt", request.getPrompt());
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
        
        try {
            Map body = callWithRetry(url, entity, Map.class, actionType);
            if (body != null && Boolean.TRUE.equals(body.get("success"))) {
                String textResponse = (String) body.get("text");
                saveAiRequest(user, actionType, request.getPrompt(), textResponse);
                return textResponse;
            } else {
                String errorMsg = body != null && body.containsKey("detail") ? (String) body.get("detail") : "Unknown Python engine error";
                throw new RuntimeException("Python AI Engine returned failure: " + errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to generate content from AI: " + e.getMessage(), e);
        }
    }

    public String generateGhibli(AiRequestDto request) {
        User user = userService.getCurrentUser();
        
        String url = pythonEngineUrl + "/api/python/ghibli";
        
        String ghibliPrompt = request.getPrompt() != null && !request.getPrompt().isEmpty() 
            ? request.getPrompt() 
            : "Transform this image into a beautiful Studio Ghibli anime scene. Use soft lighting, pastel colors, hand-drawn textures, and a whimsical atmosphere.";
            
        // Log telemetry
        if (request.getImage() != null) {
            System.out.println("[AiService] Received Ghibli request. Base64 payload length: " + request.getImage().length());
            try {
                byte[] decoded = java.util.Base64.getDecoder().decode(request.getImage());
                java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
                byte[] hash = digest.digest(decoded);
                StringBuilder hexString = new StringBuilder();
                for (byte b : hash) {
                    String hex = Integer.toHexString(0xff & b);
                    if (hex.length() == 1) hexString.append('0');
                    hexString.append(hex);
                }
                System.out.println("[AiService] Decoded image size: " + decoded.length + " bytes");
                System.out.println("[AiService] Input image SHA-256 hash: " + hexString.toString());
            } catch (Exception e) {
                System.err.println("[AiService] Error calculating SHA-256 hash: " + e.getMessage());
            }
        }
            
        Map<String, Object> payload = new HashMap<>();
        payload.put("prompt", ghibliPrompt);
        payload.put("image", request.getImage());
        payload.put("strength", request.getStrength() != null ? request.getStrength() : 0.60);
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
        
        try {
            Map body = callWithRetry(url, entity, Map.class, "GHIBLI");
            if (body != null && Boolean.TRUE.equals(body.get("success"))) {
                String imageResponse = (String) body.get("image");
                if (imageResponse != null) {
                    System.out.println("[AiService] Received Ghibli response. Base64 payload length: " + imageResponse.length());
                }
                saveAiRequest(user, "GHIBLI", ghibliPrompt, "Image generation simulated");
                return imageResponse;
            } else {
                String errorMsg = body != null && body.containsKey("detail") ? (String) body.get("detail") : "Unknown Python engine error";
                throw new RuntimeException("Python AI Engine returned failure: " + errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to process Ghibli transformation: " + e.getMessage(), e);
        }
    }
    
    public String removeBg(AiRequestDto request) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image", request.getImage());
        return callPythonAi("/api/python/edit/remove-bg", payload, "REMOVE_BG", "Remove background", "Image background removed");
    }

    public String replaceBg(AiRequestDto request) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image", request.getImage());
        payload.put("prompt", request.getPrompt());
        return callPythonAi("/api/python/edit/bg-replace", payload, "BG_REPLACE", request.getPrompt(), "Background replaced");
    }

    public Map<String, Object> inpaint(AiRequestDto request) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image", request.getImage());
        payload.put("prompt", request.getPrompt());
        payload.put("mask", request.getMask());
        payload.put("mode", request.getMode());
        
        User user = userService.getCurrentUser();
        String url = pythonEngineUrl + "/api/python/edit/inpaint";
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
        
        try {
            Map body = callWithRetry(url, entity, Map.class, "INPAINT");
            if (body != null && Boolean.TRUE.equals(body.get("success"))) {
                String imageResponse = (String) body.get("image");
                saveAiRequest(user, "INPAINT", request.getPrompt(), "Object removed/replaced");
                
                Map<String, Object> response = new HashMap<>();
                response.put("image", imageResponse);
                response.put("model_used", body.get("model_used"));
                response.put("processing_time", body.get("processing_time"));
                response.put("image_size", body.get("image_size"));
                return response;
            } else {
                String errorMsg = body != null && body.containsKey("detail") ? (String) body.get("detail") : "Unknown Python engine error";
                throw new RuntimeException("Python AI Engine returned failure: " + errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to process INPAINT: " + e.getMessage(), e);
        }
    }

    public String upscale(AiRequestDto request) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image", request.getImage());
        payload.put("upscale_factor", request.getUpscaleFactor() != null ? request.getUpscaleFactor() : 2);
        return callPythonAi("/api/python/edit/upscale", payload, "UPSCALE", "Upscale image", "Image resolution increased");
    }

    public String enhanceSketch(AiRequestDto request) {
        Map<String, Object> payload = new HashMap<>();
        payload.put("image", request.getImage());
        return callPythonAi("/api/python/edit/sketch", payload, "SKETCH", "Enhance sketch", "Sketch lines sharpened");
    }

    private String callPythonAi(String endpoint, Map<String, Object> payload, String actionType, String prompt, String details) {
        User user = userService.getCurrentUser();
        String url = pythonEngineUrl + endpoint;
        
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(payload, headers);
        
        try {
            Map body = callWithRetry(url, entity, Map.class, actionType);
            if (body != null && Boolean.TRUE.equals(body.get("success"))) {
                String imageResponse = (String) body.get("image");
                saveAiRequest(user, actionType, prompt, details);
                return imageResponse;
            } else {
                String errorMsg = body != null && body.containsKey("detail") ? (String) body.get("detail") : "Unknown Python engine error";
                throw new RuntimeException("Python AI Engine returned failure: " + errorMsg);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to process " + actionType + ": " + e.getMessage(), e);
        }
    }

    private void saveAiRequest(User user, String actionType, String prompt, String result) {
        AiRequest aiReq = AiRequest.builder()
                .user(user)
                .actionType(actionType)
                .promptUsed(prompt)
                .resultUrlOrText(result)
                .build();
        aiRequestRepository.save(aiReq);
    }
}
