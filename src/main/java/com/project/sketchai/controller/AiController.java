package com.project.sketchai.controller;

import com.project.sketchai.dto.AiRequestDto;
import com.project.sketchai.service.AiService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
public class AiController {

    private final AiService aiService;

    @PostMapping("/generate")
    public ResponseEntity<Map<String, String>> generate(@RequestBody AiRequestDto request) {
        String result = aiService.generateCaptionOrSuggestion(request, "TEXT_GENERATION");
        Map<String, String> response = new HashMap<>();
        response.put("text", result);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/ghibli")
    public ResponseEntity<Map<String, String>> ghibli(@RequestBody AiRequestDto request) {
        String base64Image = aiService.generateGhibli(request);
        Map<String, String> response = new HashMap<>();
        response.put("image", base64Image);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/remove-bg")
    public ResponseEntity<Map<String, String>> removeBg(@RequestBody AiRequestDto request) {
        String base64Image = aiService.removeBg(request);
        Map<String, String> response = new HashMap<>();
        response.put("image", base64Image);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/replace-bg")
    public ResponseEntity<Map<String, String>> replaceBg(@RequestBody AiRequestDto request) {
        String base64Image = aiService.replaceBg(request);
        Map<String, String> response = new HashMap<>();
        response.put("image", base64Image);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/inpaint")
    public ResponseEntity<Map<String, Object>> inpaint(@RequestBody AiRequestDto request) {
        Map<String, Object> response = aiService.inpaint(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/upscale")
    public ResponseEntity<Map<String, String>> upscale(@RequestBody AiRequestDto request) {
        String base64Image = aiService.upscale(request);
        Map<String, String> response = new HashMap<>();
        response.put("image", base64Image);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/sketch-enhance")
    public ResponseEntity<Map<String, String>> sketchEnhance(@RequestBody AiRequestDto request) {
        String base64Image = aiService.enhanceSketch(request);
        Map<String, String> response = new HashMap<>();
        response.put("image", base64Image);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/strategy")
    public ResponseEntity<Map<String, String>> strategy(@RequestBody AiRequestDto request) {
        String result = aiService.generateCaptionOrSuggestion(request, "STRATEGY");
        Map<String, String> response = new HashMap<>();
        response.put("text", result);
        return ResponseEntity.ok(response);
    }
}
