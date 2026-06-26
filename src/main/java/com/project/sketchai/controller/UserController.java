package com.project.sketchai.controller;

import com.project.sketchai.dto.ApiResponse;
import com.project.sketchai.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/user")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;

    @GetMapping("/profile")
    public ResponseEntity<ApiResponse> getProfile() {
        return ResponseEntity.ok(ApiResponse.builder()
                .success(true)
                .data(userService.getUserProfile())
                .build());
    }

    @GetMapping("/dashboard")
    public ResponseEntity<ApiResponse> getDashboardStats() {
        return ResponseEntity.ok(ApiResponse.builder()
                .success(true)
                .data(userService.getDashboardStats())
                .build());
    }
}
